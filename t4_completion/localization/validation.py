"""Fail-closed validation for a coordinator-run localization Oracle result."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .contracts import (
    ALL_SOURCES,
    CANONICAL_CHILD_FRAME,
    CANONICAL_ODOMETRY_TOPIC,
    CANONICAL_PARENT_FRAME,
    CANONICAL_TF_TOPIC,
    CUVSLAM,
    ISAAC_GT,
    LIDAR_IMU,
    Pose,
)


RECORD_KEYS = {
    "schema_version",
    "event",
    "runtime_policy",
    "runtime_target",
    "generation",
    "decision_monotonic_ns",
    "previous_source",
    "selected_source",
    "switch_reason",
    "switch_event",
    "source_health",
    "output",
    "odometry_deferred",
    "deviation",
    "switch_translation_jump_m",
    "switch_rotation_jump_rad",
}
SOURCE_HEALTH_KEYS = {
    "source",
    "configured",
    "state",
    "reason",
    "generation",
    "last_stamp_ns",
    "age_sec",
    "stamp_age_sec",
    "valid_count",
    "invalid_count",
    "ordered_count",
    "backend_ready",
    "consecutive_healthy_decisions",
}
OUTPUT_KEYS = {
    "source",
    "generation",
    "sequence_id",
    "stamp_ns",
    "parent_frame",
    "child_frame",
    "pose",
    "linear_velocity_xyz",
    "angular_velocity_xyz",
    "odometry_topic",
    "tf_topic",
}
DEVIATION_KEYS = {
    "schema_version",
    "severity",
    "code",
    "runtime_policy",
    "runtime_target",
    "fallback_source",
    "sensor_odometry_status",
    "sensor_sources",
}
SWITCH_REASONS = {
    "INITIAL_PREFERRED_HEALTHY",
    "INITIAL_ALTERNATE_HEALTHY",
    "ALL_SENSOR_ODOMETRY_DEFERRED",
    "STRICT_SENSOR_ODOMETRY_UNAVAILABLE_FAIL_CLOSED",
    "NO_HEALTHY_SOURCE_FAIL_CLOSED",
    "RECOVERED_AFTER_HYSTERESIS",
    "ACTIVE_HEALTHY",
    "ACTIVE_SOURCE_UNHEALTHY",
    "GT_UNAVAILABLE_FAIL_CLOSED",
    "CANDIDATE_TIMESTAMP_NOT_NEWER",
}


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _decode(raw: str, label: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}_invalid_json") from exc


def _load(path: Path) -> dict[str, Any]:
    value = _decode(path.read_text(encoding="utf-8"), path.name)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}_must_be_object")
    return value


def _exact(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label}_keys_mismatch:missing={sorted(keys-set(value))}:"
            f"unknown={sorted(set(value)-keys)}"
        )


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}_must_be_nonnegative_integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}_must_be_number")
    converted = float(value)
    if not (-float("inf") < converted < float("inf")):
        raise ValueError(f"{label}_must_be_finite")
    return converted


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}_must_be_length_{length}_array")
    return tuple(_finite_number(item, label) for item in value)


def _optional_source(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in ALL_SOURCES:
        raise ValueError(f"{label}_invalid")
    return value


def _validate_deviation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}_must_be_object")
    _exact(value, DEVIATION_KEYS, label)
    if (
        _nonnegative_integer(value["schema_version"], f"{label}_schema_version")
        != 1
        or value["severity"] != "WARN"
        or value["code"] != "ODOMETRY_DEFERRED_ISAAC_GT_FALLBACK"
        or value["runtime_policy"] != "completion_sim"
        or value["runtime_target"] != "isaac_simulation"
        or value["fallback_source"] != ISAAC_GT
        or value["sensor_odometry_status"] != "STRICT_EXTENSION_PENDING"
    ):
        raise ValueError(f"{label}_identity_changed")
    sensor_sources = value["sensor_sources"]
    if not isinstance(sensor_sources, dict) or set(sensor_sources) != {
        CUVSLAM,
        LIDAR_IMU,
    }:
        raise ValueError(f"{label}_sensor_sources_invalid")
    for source, source_health in sensor_sources.items():
        if not isinstance(source_health, dict) or set(source_health) != {
            "state",
            "reason",
        }:
            raise ValueError(f"{label}_source_health_invalid:{source}")
        if (
            not isinstance(source_health["state"], str)
            or not source_health["state"]
            or not isinstance(source_health["reason"], str)
            or not source_health["reason"]
        ):
            raise ValueError(f"{label}_source_health_empty:{source}")
    return value


def _record_aggregates(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("localization_records_must_be_nonempty_jsonl")
    selections: dict[str, int] = {}
    output_count = 0
    switch_count = 0
    deferred_count = 0
    deviation_count = 0
    maximum_translation = 0.0
    maximum_rotation = 0.0
    representative_deviations: dict[str, dict[str, Any]] = {}
    previous_decision_ns: int | None = None
    previous_generation: int | None = None
    previous_selected: str | None = None
    previous_output_stamp: int | None = None
    previous_output_pose: Pose | None = None
    last_record: dict[str, Any] | None = None
    for index, line in enumerate(lines):
        label = f"localization_record_{index}"
        record = _decode(line, label)
        if not isinstance(record, dict):
            raise ValueError(f"{label}_must_be_object")
        _exact(record, RECORD_KEYS, label)
        if (
            _nonnegative_integer(record["schema_version"], f"{label}_schema") != 1
            or record["event"] != "localization_selection"
            or record["runtime_policy"] != "completion_sim"
            or record["runtime_target"] != "isaac_simulation"
        ):
            raise ValueError(f"{label}_identity_changed")
        generation = _nonnegative_integer(record["generation"], f"{label}_generation")
        decision_ns = _nonnegative_integer(
            record["decision_monotonic_ns"], f"{label}_decision_monotonic_ns"
        )
        if previous_decision_ns is not None and decision_ns <= previous_decision_ns:
            raise ValueError(f"{label}_decision_time_not_strictly_monotonic")
        if previous_generation is not None and generation < previous_generation:
            raise ValueError(f"{label}_generation_regression")
        generation_boundary = (
            previous_generation is None or generation > previous_generation
        )
        if generation_boundary:
            previous_selected = None
            previous_output_stamp = None
            previous_output_pose = None
        previous_source = _optional_source(
            record["previous_source"], f"{label}_previous_source"
        )
        selected_source = _optional_source(
            record["selected_source"], f"{label}_selected_source"
        )
        if generation_boundary:
            if previous_source is not None:
                raise ValueError(f"{label}_generation_starts_with_previous_source")
        elif previous_source != previous_selected:
            raise ValueError(f"{label}_previous_source_chain_broken")
        if not isinstance(record["switch_event"], bool):
            raise ValueError(f"{label}_switch_event_not_boolean")
        if record["switch_event"] is not (previous_source != selected_source):
            raise ValueError(f"{label}_switch_event_inconsistent")
        if record["switch_reason"] not in SWITCH_REASONS:
            raise ValueError(f"{label}_switch_reason_unknown")
        if not isinstance(record["odometry_deferred"], bool) or record[
            "odometry_deferred"
        ] is not (selected_source == ISAAC_GT):
            raise ValueError(f"{label}_deferred_state_inconsistent")
        source_health = record["source_health"]
        if not isinstance(source_health, dict) or set(source_health) != set(ALL_SOURCES):
            raise ValueError(f"{label}_source_health_set_changed")
        for source, health in source_health.items():
            if not isinstance(health, dict):
                raise ValueError(f"{label}_{source}_health_not_object")
            _exact(health, SOURCE_HEALTH_KEYS, f"{label}_{source}_health")
            if (
                health["source"] != source
                or not isinstance(health["configured"], bool)
                or not isinstance(health["backend_ready"], bool)
                or not isinstance(health["state"], str)
                or not health["state"]
                or not isinstance(health["reason"], str)
                or not health["reason"]
                or _nonnegative_integer(
                    health["generation"], f"{label}_{source}_health_generation"
                )
                != generation
            ):
                raise ValueError(f"{label}_{source}_health_identity_invalid")
            for count_key in (
                "valid_count",
                "invalid_count",
                "ordered_count",
                "consecutive_healthy_decisions",
            ):
                _nonnegative_integer(
                    health[count_key], f"{label}_{source}_{count_key}"
                )
            if health["last_stamp_ns"] is not None:
                if (
                    _nonnegative_integer(
                        health["last_stamp_ns"], f"{label}_{source}_last_stamp_ns"
                    )
                    <= 0
                ):
                    raise ValueError(f"{label}_{source}_last_stamp_nonpositive")
            for age_key in ("age_sec", "stamp_age_sec"):
                if health[age_key] is not None:
                    _finite_number(health[age_key], f"{label}_{source}_{age_key}")
        output = record["output"]
        output_pose: Pose | None = None
        if output is not None:
            if not isinstance(output, dict):
                raise ValueError(f"{label}_output_not_object")
            _exact(output, OUTPUT_KEYS, f"{label}_output")
            if (
                output["source"] != selected_source
                or _nonnegative_integer(
                    output["generation"], f"{label}_output_generation"
                )
                != generation
                or output["parent_frame"] != CANONICAL_PARENT_FRAME
                or output["child_frame"] != CANONICAL_CHILD_FRAME
                or output["odometry_topic"] != CANONICAL_ODOMETRY_TOPIC
                or output["tf_topic"] != CANONICAL_TF_TOPIC
            ):
                raise ValueError(f"{label}_canonical_output_identity_changed")
            _nonnegative_integer(output["sequence_id"], f"{label}_output_sequence")
            output_stamp = _nonnegative_integer(
                output["stamp_ns"], f"{label}_output_stamp"
            )
            if output_stamp <= 0 or (
                previous_output_stamp is not None
                and output_stamp <= previous_output_stamp
            ):
                raise ValueError(f"{label}_output_stamp_not_strictly_monotonic")
            pose = output["pose"]
            if not isinstance(pose, dict) or set(pose) != {
                "translation_xyz",
                "quaternion_xyzw",
            }:
                raise ValueError(f"{label}_output_pose_invalid")
            _vector(pose["translation_xyz"], 3, f"{label}_translation")
            quaternion = _vector(
                pose["quaternion_xyzw"], 4, f"{label}_quaternion"
            )
            if abs(math.sqrt(sum(value * value for value in quaternion)) - 1.0) > 0.001:
                raise ValueError(f"{label}_output_quaternion_not_unit")
            _vector(output["linear_velocity_xyz"], 3, f"{label}_linear_velocity")
            _vector(output["angular_velocity_xyz"], 3, f"{label}_angular_velocity")
            output_pose = Pose(
                tuple(float(value) for value in pose["translation_xyz"]),  # type: ignore[arg-type]
                quaternion,  # type: ignore[arg-type]
            ).normalized()
            previous_output_stamp = output_stamp
            output_count += 1
        translation_jump = record["switch_translation_jump_m"]
        rotation_jump = record["switch_rotation_jump_rad"]
        if translation_jump is not None:
            translation_jump = _finite_number(
                translation_jump, f"{label}_switch_translation_jump_m"
            )
            if translation_jump < 0:
                raise ValueError(f"{label}_negative_translation_jump")
            maximum_translation = max(maximum_translation, translation_jump)
        if rotation_jump is not None:
            rotation_jump = _finite_number(
                rotation_jump, f"{label}_switch_rotation_jump_rad"
            )
            if rotation_jump < 0:
                raise ValueError(f"{label}_negative_rotation_jump")
            maximum_rotation = max(maximum_rotation, rotation_jump)
        switch_with_continuity = (
            record["switch_event"]
            and output_pose is not None
            and previous_output_pose is not None
        )
        if switch_with_continuity:
            if translation_jump is None or rotation_jump is None:
                raise ValueError(f"{label}_switch_jump_measurement_missing")
            recomputed_translation = output_pose.translation_distance(
                previous_output_pose
            )
            recomputed_rotation = output_pose.rotation_distance_rad(
                previous_output_pose
            )
            if (
                abs(translation_jump - recomputed_translation) > 1e-12
                or abs(rotation_jump - recomputed_rotation) > 1e-12
            ):
                raise ValueError(f"{label}_switch_jump_measurement_mismatch")
        elif translation_jump is not None or rotation_jump is not None:
            raise ValueError(f"{label}_unexpected_switch_jump_measurement")
        deviation = record["deviation"]
        if selected_source == ISAAC_GT:
            if deviation is not None:
                validated = _validate_deviation(deviation, f"{label}_deviation")
                key = json.dumps(validated, sort_keys=True, allow_nan=False)
                representative_deviations.setdefault(key, validated)
                deviation_count += 1
        elif deviation is not None:
            raise ValueError(f"{label}_unexpected_deviation")
        if selected_source is not None:
            selections[selected_source] = selections.get(selected_source, 0) + 1
        switch_count += int(record["switch_event"])
        deferred_count += int(record["odometry_deferred"])
        previous_decision_ns = decision_ns
        previous_generation = generation
        previous_selected = selected_source
        if output_pose is not None:
            previous_output_pose = output_pose
        last_record = record
    assert last_record is not None
    return {
        "decision_count": len(lines),
        "output_count": output_count,
        "switch_count": switch_count,
        "selection_counts": dict(sorted(selections.items())),
        "deferred_count": deferred_count,
        "deviation_count": deviation_count,
        "maximum_translation": maximum_translation,
        "maximum_rotation": maximum_rotation,
        "representative_deviations": list(representative_deviations.values()),
        "last_record": last_record,
    }


def validate_oracle_result(result_dir: Path, gate_path: Path) -> dict[str, Any]:
    gate = _load(gate_path)
    _exact(
        gate,
        {
            "schema_version",
            "profile",
            "runtime_target",
            "episode_count",
            "minimum_success_count",
            "maximum_collision_count",
            "maximum_fall_count",
            "maximum_generation_contamination_count",
            "maximum_switch_translation_jump_m",
            "maximum_switch_rotation_jump_rad",
            "require_deviation_when_gt_selected",
            "fallback_disposition",
            "sensor_odometry_fallback_status",
            "require_fresh_result_dir",
        },
        "gate",
    )
    if (
        _nonnegative_integer(gate["schema_version"], "gate_schema_version") != 1
        or gate["profile"] != "completion_sim"
        or gate["runtime_target"] != "isaac_simulation"
        or _nonnegative_integer(gate["episode_count"], "gate_episode_count") != 10
        or _nonnegative_integer(
            gate["minimum_success_count"], "gate_minimum_success_count"
        )
        != 8
        or _nonnegative_integer(
            gate["maximum_collision_count"], "gate_maximum_collision_count"
        )
        != 0
        or _nonnegative_integer(
            gate["maximum_fall_count"], "gate_maximum_fall_count"
        )
        != 0
        or _nonnegative_integer(
            gate["maximum_generation_contamination_count"],
            "gate_maximum_generation_contamination_count",
        )
        != 0
        or _finite_number(
            gate["maximum_switch_translation_jump_m"],
            "gate_maximum_switch_translation_jump_m",
        )
        != 1e-9
        or _finite_number(
            gate["maximum_switch_rotation_jump_rad"],
            "gate_maximum_switch_rotation_jump_rad",
        )
        != 1e-9
        or gate["fallback_disposition"] != "PASS_WITH_DEVIATION"
        or gate["sensor_odometry_fallback_status"] != "STRICT_EXTENSION_PENDING"
        or gate["require_deviation_when_gt_selected"] is not True
        or gate["require_fresh_result_dir"] is not True
    ):
        raise ValueError("oracle_gate_contract_changed")

    selector = _load(result_dir / "localization_selector/localization_summary.json")
    deviations = _load(
        result_dir / "localization_selector/localization_deviations.json"
    )
    episodes = _load(result_dir / "oracle_episode_summary.json")
    cleanup = _load(result_dir / "localization_cleanup.json")
    records = _record_aggregates(
        result_dir / "localization_selector/localization_records.jsonl"
    )
    _exact(
        selector,
        {
            "schema_version",
            "status",
            "decision_count",
            "output_count",
            "switch_count",
            "selection_counts",
            "odometry_deferred_decision_count",
            "maximum_switch_translation_jump_m",
            "maximum_switch_rotation_jump_rad",
            "selector",
            "deviation_artifact",
        },
        "selector_summary",
    )
    _exact(
        deviations,
        {
            "schema_version",
            "status",
            "odometry_deferred",
            "deviation_count",
            "deviations",
        },
        "deviations",
    )
    _exact(
        episodes,
        {
            "schema_version",
            "status",
            "episode_count",
            "success_count",
            "collision_count",
            "fall_count",
            "generation_contamination_count",
            "stale_motion_count",
        },
        "episodes",
    )
    _exact(
        cleanup,
        {"schema_version", "status", "pid_count", "pgid_count", "socket_count"},
        "cleanup",
    )
    selector_state = selector.get("selector")
    if not isinstance(selector_state, dict):
        raise ValueError("selector_state_must_be_object")
    _exact(
        selector_state,
        {
            "schema_version",
            "runtime_policy",
            "runtime_target",
            "generation",
            "selected_source",
            "current_output_available",
            "switch_count",
            "deviation_count",
            "no_output_count",
            "odometry_deferred",
            "sensor_odometry_status",
        },
        "selector_state",
    )
    if (
        _nonnegative_integer(selector["schema_version"], "selector_schema_version")
        != 1
        or _nonnegative_integer(deviations["schema_version"], "deviation_schema_version")
        != 1
        or _nonnegative_integer(episodes["schema_version"], "episodes_schema_version")
        != 1
        or _nonnegative_integer(cleanup["schema_version"], "cleanup_schema_version")
        != 1
        or _nonnegative_integer(
            selector_state["schema_version"], "selector_state_schema_version"
        )
        != 1
    ):
        raise ValueError("online_artifact_schema_version_changed")
    selection_counts = selector.get("selection_counts")
    if not isinstance(selection_counts, dict) or any(
        not isinstance(name, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in selection_counts.items()
    ):
        raise ValueError("selector_selection_counts_invalid")
    if not set(selection_counts).issubset(ALL_SOURCES):
        raise ValueError("selector_selection_counts_unknown_source")
    gt_count = int(selection_counts.get(ISAAC_GT, 0))
    selected_source = selector_state.get("selected_source")
    if not isinstance(selected_source, str) or selected_source not in ALL_SOURCES:
        raise ValueError("selector_selected_source_invalid")
    if not isinstance(selector_state.get("current_output_available"), bool) or not isinstance(
        selector_state.get("odometry_deferred"), bool
    ):
        raise ValueError("selector_state_booleans_invalid")
    selector_generation = _nonnegative_integer(
        selector_state.get("generation"), "selector_state_generation"
    )
    selector_switch_count = _nonnegative_integer(
        selector_state.get("switch_count"), "selector_state_switch_count"
    )
    selector_deviation_count = _nonnegative_integer(
        selector_state.get("deviation_count"), "selector_state_deviation_count"
    )
    _nonnegative_integer(
        selector_state.get("no_output_count"), "selector_state_no_output_count"
    )
    deviation_records = deviations.get("deviations")
    if not isinstance(deviation_records, list):
        raise ValueError("deviation_records_must_be_array")
    explicit_gt_deviation = False
    for index, record in enumerate(deviation_records):
        _validate_deviation(record, f"deviation_record_{index}")
        explicit_gt_deviation = True
    decision_count = _nonnegative_integer(
        selector.get("decision_count"), "selector_decision_count"
    )
    output_count = _nonnegative_integer(
        selector.get("output_count"), "selector_output_count"
    )
    deferred_count = _nonnegative_integer(
        selector.get("odometry_deferred_decision_count"),
        "selector_deferred_decision_count",
    )
    deviation_count = _nonnegative_integer(
        deviations.get("deviation_count"), "deviation_count"
    )
    outer_switch_count = _nonnegative_integer(
        selector.get("switch_count"), "selector_switch_count"
    )
    if not isinstance(deviations.get("odometry_deferred"), bool):
        raise ValueError("deviation_odometry_deferred_must_be_boolean")
    if not (
        (deviation_count == 0 and deviation_records == [])
        or (deviation_count > 0 and 1 <= len(deviation_records) <= deviation_count)
    ):
        raise ValueError("deviation_record_count_inconsistent")
    gt_deviation_ok = (
        gt_count == 0
        or (
            selector.get("status") == "PASS_WITH_DEVIATION"
            and deviations.get("status") == "WARN"
            and deviations.get("odometry_deferred") is True
            and deviation_count > 0
            and explicit_gt_deviation
            and gt_count == deferred_count == deviation_count
            and selector_state.get("sensor_odometry_status")
            == "STRICT_EXTENSION_PENDING"
        )
    )
    checks = {
        "profile_and_target": (
            selector_state.get("runtime_policy") == "completion_sim"
            and selector_state.get("runtime_target")
            == "isaac_simulation"
        ),
        "current_output_available": selector_state.get("current_output_available")
        is True,
        "selector_status": selector.get("status")
        == ("PASS_WITH_DEVIATION" if gt_count else "PASS"),
        "selector_output_present": output_count > 0 and output_count <= decision_count,
        "artifact_identity": selector.get("deviation_artifact")
        == "localization_deviations.json",
        "selection_count_consistent": (
            output_count <= sum(selection_counts.values()) <= decision_count
        ),
        "raw_record_recomputation": (
            records["decision_count"] == decision_count
            and records["output_count"] == output_count
            and records["switch_count"] == outer_switch_count
            and records["selection_counts"] == dict(sorted(selection_counts.items()))
            and records["deferred_count"] == deferred_count
            and records["deviation_count"] == deviation_count
            and records["maximum_translation"]
            == _finite_number(
                selector.get("maximum_switch_translation_jump_m"),
                "maximum_switch_translation_jump_m",
            )
            and records["maximum_rotation"]
            == _finite_number(
                selector.get("maximum_switch_rotation_jump_rad"),
                "maximum_switch_rotation_jump_rad",
            )
            and records["representative_deviations"] == deviation_records
            and records["last_record"]["generation"] == selector_generation
            and records["last_record"]["selected_source"] == selected_source
            and records["last_record"]["source_health"][selected_source]["state"]
            == "HEALTHY"
        ),
        "counter_bounds": (
            outer_switch_count <= decision_count
            and deferred_count <= decision_count
            and deviation_count <= decision_count
            and selector_switch_count == outer_switch_count
            # The evidence writer records every output plus throttled no-output
            # GT decisions.  The selector counter covers output deviations, so
            # the retained evidence count may be larger but never smaller.
            and selector_deviation_count <= deviation_count
        ),
        "selector_state_consistent": (
            selector_generation >= 0
            and selection_counts.get(selected_source, 0) > 0
            and selector_state.get("odometry_deferred")
            is (selected_source == ISAAC_GT)
            and selector_state.get("sensor_odometry_status")
            == (
                "STRICT_EXTENSION_PENDING"
                if gt_count > 0
                else "SELECTED_OR_NOT_YET_EVALUATED"
            )
        ),
        "translation_continuity": (
            0.0
            <= _finite_number(
                selector.get("maximum_switch_translation_jump_m"),
                "maximum_switch_translation_jump_m",
            )
            <= _finite_number(
                gate["maximum_switch_translation_jump_m"],
                "gate_maximum_switch_translation_jump_m",
            )
        ),
        "rotation_continuity": (
            0.0
            <= _finite_number(
                selector.get("maximum_switch_rotation_jump_rad"),
                "maximum_switch_rotation_jump_rad",
            )
            <= _finite_number(
                gate["maximum_switch_rotation_jump_rad"],
                "gate_maximum_switch_rotation_jump_rad",
            )
        ),
        "gt_fallback_deviation": gt_deviation_ok,
        "no_silent_deviation": (
            gt_count > 0
            or (
                deferred_count == 0
                and deviation_count == 0
                and deviations.get("status") == "NONE"
                and deviations.get("odometry_deferred") is False
                and deviation_records == []
                and selector_state.get("odometry_deferred") is False
            )
        ),
        "episode_status": episodes.get("status") == "PASS",
        "episode_count": _nonnegative_integer(
            episodes.get("episode_count"), "episode_count"
        )
        == _nonnegative_integer(gate["episode_count"], "gate_episode_count"),
        "success_count": _nonnegative_integer(
            episodes.get("success_count"), "success_count"
        )
        >= _nonnegative_integer(
            gate["minimum_success_count"], "gate_minimum_success_count"
        )
        and _nonnegative_integer(episodes.get("success_count"), "success_count")
        <= _nonnegative_integer(episodes.get("episode_count"), "episode_count"),
        "collision_count": _nonnegative_integer(
            episodes.get("collision_count"), "collision_count"
        )
        <= _nonnegative_integer(
            gate["maximum_collision_count"], "gate_maximum_collision_count"
        ),
        "fall_count": _nonnegative_integer(episodes.get("fall_count"), "fall_count")
        <= _nonnegative_integer(gate["maximum_fall_count"], "gate_maximum_fall_count"),
        "generation_contamination_count": _nonnegative_integer(
            episodes.get("generation_contamination_count"),
            "generation_contamination_count",
        )
        <= _nonnegative_integer(
            gate["maximum_generation_contamination_count"],
            "gate_maximum_generation_contamination_count",
        ),
        "stale_motion_count": _nonnegative_integer(
            episodes.get("stale_motion_count"), "stale_motion_count"
        )
        == 0,
        "cleanup": (
            cleanup.get("status") == "PASS"
            and _nonnegative_integer(cleanup.get("pid_count"), "cleanup_pid_count")
            == 0
            and _nonnegative_integer(
                cleanup.get("pgid_count"), "cleanup_pgid_count"
            )
            == 0
            and _nonnegative_integer(
                cleanup.get("socket_count"), "cleanup_socket_count"
            )
            == 0
        ),
    }
    errors = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "functional_disposition": (
            "PASS_WITH_DEVIATION" if not errors and gt_count else "PASS"
        )
        if not errors
        else "FAIL",
        "sensor_odometry_status": (
            "STRICT_EXTENSION_PENDING" if gt_count else "OBSERVED_WITHOUT_GT_FALLBACK"
        ),
        "gt_fallback_selection_count": gt_count,
        "checks": checks,
        "errors": errors,
    }


def write_validation(result_dir: Path, gate_path: Path) -> dict[str, Any]:
    output = result_dir / "localization_validation.json"
    try:
        with output.open("x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        raise FileExistsError(
            "refusing_to_replace_localization_validation"
        ) from exc
    temporary = output.with_suffix(".json.tmp")
    temporary_created = False
    try:
        payload = validate_oracle_result(result_dir, gate_path)
        with temporary.open("x", encoding="utf-8") as stream:
            temporary_created = True
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
        os.replace(temporary, output)
    except BaseException:
        try:
            if temporary_created:
                temporary.unlink(missing_ok=True)
        finally:
            output.unlink(missing_ok=True)
        raise
    return payload
