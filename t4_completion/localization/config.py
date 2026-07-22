"""Fail-closed loader for the completion-only localization configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ALL_SOURCES,
    CANONICAL_CHILD_FRAME,
    CANONICAL_ODOMETRY_TOPIC,
    CANONICAL_PARENT_FRAME,
    CANONICAL_TF_TOPIC,
    CUVSLAM,
    DEVIATION_TOPIC,
    ISAAC_GT,
    LIDAR_IMU,
    SOURCE_HEALTH_TOPIC,
    SWITCH_EVENT_TOPIC,
)


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}_must_be_object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label}_keys_mismatch:missing={sorted(expected-actual)}:"
            f"unknown={sorted(actual-expected)}"
        )
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label}_must_be_boolean")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}_must_be_number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label}_must_be_finite")
    return converted


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}_must_be_integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}_must_be_nonempty_string")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{label}_must_be_string_array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    role: str
    priority: int
    enabled: bool
    odometry_topic: str
    health_topic: str
    expected_parent_frame: str
    expected_child_frame: str
    startup_state: str
    startup_reason: str


@dataclass(frozen=True, slots=True)
class SelectorPolicy:
    runtime_policy: str
    runtime_target: str
    source_timeout_ns: int
    tf_tolerance_ns: int
    maximum_future_stamp_ns: int
    preferred_recovery_hold_ns: int
    minimum_dwell_ns: int
    quaternion_norm_tolerance: float
    allow_isaac_gt_fallback: bool

    def __post_init__(self) -> None:
        if self.runtime_policy == "completion_sim":
            if self.runtime_target != "isaac_simulation":
                raise ValueError("completion_sim_requires_isaac_simulation_target")
            if not self.allow_isaac_gt_fallback:
                raise ValueError("completion_sim_requires_explicit_gt_fallback_contract")
            if self.source_timeout_ns != 5_000_000_000:
                raise ValueError("completion_sim_source_timeout_must_equal_5_sec")
            if self.tf_tolerance_ns != 2_500_000_000:
                raise ValueError("completion_sim_tf_tolerance_must_equal_2_5_sec")
            if self.maximum_future_stamp_ns != 1_000_000:
                raise ValueError("completion_sim_future_stamp_must_equal_1_ms")
        elif self.runtime_policy == "strict_evidence":
            if self.allow_isaac_gt_fallback:
                raise ValueError("strict_evidence_forbids_gt_fallback")
            if self.source_timeout_ns != 350_000_000:
                raise ValueError("strict_evidence_source_timeout_must_equal_0_35_sec")
            if self.tf_tolerance_ns != 0:
                raise ValueError("strict_evidence_tf_tolerance_must_equal_zero")
            if self.maximum_future_stamp_ns != 0:
                raise ValueError("strict_evidence_future_stamp_must_equal_zero")
        else:
            raise ValueError("unknown_runtime_policy")
        if self.runtime_target in {"real_go2", "hardware_motion"}:
            raise ValueError("localization_completion_fallback_forbidden_on_hardware")
        if self.preferred_recovery_hold_ns < 0 or self.minimum_dwell_ns < 0:
            raise ValueError("negative_hysteresis_not_allowed")
        if not 0.0 < self.quaternion_norm_tolerance <= 0.1:
            raise ValueError("invalid_quaternion_norm_tolerance")

    @classmethod
    def strict_evidence(cls, *, runtime_target: str = "isaac_simulation") -> "SelectorPolicy":
        return cls(
            runtime_policy="strict_evidence",
            runtime_target=runtime_target,
            source_timeout_ns=350_000_000,
            tf_tolerance_ns=0,
            maximum_future_stamp_ns=0,
            preferred_recovery_hold_ns=1_000_000_000,
            minimum_dwell_ns=500_000_000,
            quaternion_norm_tolerance=0.001,
            allow_isaac_gt_fallback=False,
        )


@dataclass(frozen=True, slots=True)
class LocalizationConfig:
    schema_version: int
    policy: SelectorPolicy
    allowed_targets: tuple[str, ...]
    forbidden_targets: tuple[str, ...]
    use_sim_time: bool
    sources: dict[str, SourceSpec]
    output_health_topic: str
    output_switch_topic: str
    output_deviation_topic: str
    oracle_completion_topic: str
    evidence_record_filename: str
    evidence_summary_filename: str


def _positive_topic(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "//" in value:
        raise ValueError(f"invalid_{label}")
    return value


def load_config(
    path: Path | str,
    *,
    expected_runtime_policy: str,
    expected_runtime_target: str,
) -> LocalizationConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = _exact_keys(
        payload,
        {
            "schema_version",
            "runtime_policy",
            "runtime_target",
            "allowed_targets",
            "forbidden_targets",
            "use_sim_time",
            "sources",
            "selection",
            "output",
            "evidence",
        },
        "config",
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported_config_schema")
    if root["runtime_policy"] != expected_runtime_policy:
        raise ValueError("runtime_policy_mismatch")
    if root["runtime_target"] != expected_runtime_target:
        raise ValueError("runtime_target_mismatch")
    if not isinstance(root["runtime_policy"], str) or not isinstance(
        root["runtime_target"], str
    ):
        raise ValueError("runtime_policy_and_target_must_be_strings")
    allowed = _string_tuple(root["allowed_targets"], "allowed_targets")
    forbidden = _string_tuple(root["forbidden_targets"], "forbidden_targets")
    if allowed != ("isaac_simulation",):
        raise ValueError("allowed_targets_must_be_isaac_only")
    if forbidden != ("real_go2", "hardware_motion"):
        raise ValueError("hardware_forbidden_targets_not_frozen")
    if expected_runtime_target not in allowed or expected_runtime_target in forbidden:
        raise ValueError("runtime_target_not_allowed")
    if _boolean(root["use_sim_time"], "use_sim_time") is not True:
        raise ValueError("completion_localization_requires_sim_time")

    selection = _exact_keys(
        root["selection"],
        {
            "source_timeout_sec",
            "tf_tolerance_sec",
            "maximum_future_stamp_sec",
            "preferred_recovery_hold_sec",
            "minimum_dwell_sec",
            "quaternion_norm_tolerance",
            "allow_isaac_gt_fallback",
        },
        "selection",
    )
    policy = SelectorPolicy(
        runtime_policy=str(root["runtime_policy"]),
        runtime_target=str(root["runtime_target"]),
        source_timeout_ns=int(_number(selection["source_timeout_sec"], "source_timeout_sec") * 1e9),
        tf_tolerance_ns=int(_number(selection["tf_tolerance_sec"], "tf_tolerance_sec") * 1e9),
        maximum_future_stamp_ns=int(
            _number(selection["maximum_future_stamp_sec"], "maximum_future_stamp_sec")
            * 1e9
        ),
        preferred_recovery_hold_ns=int(
            _number(selection["preferred_recovery_hold_sec"], "preferred_recovery_hold_sec") * 1e9
        ),
        minimum_dwell_ns=int(_number(selection["minimum_dwell_sec"], "minimum_dwell_sec") * 1e9),
        quaternion_norm_tolerance=_number(
            selection["quaternion_norm_tolerance"], "quaternion_norm_tolerance"
        ),
        allow_isaac_gt_fallback=_boolean(
            selection["allow_isaac_gt_fallback"], "allow_isaac_gt_fallback"
        ),
    )

    raw_sources = root["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != 3:
        raise ValueError("exactly_three_sources_required")
    sources: dict[str, SourceSpec] = {}
    source_keys = {
        "name",
        "role",
        "priority",
        "enabled",
        "odometry_topic",
        "health_topic",
        "expected_parent_frame",
        "expected_child_frame",
        "startup_state",
        "startup_reason",
    }
    for index, raw_source in enumerate(raw_sources):
        source = _exact_keys(raw_source, source_keys, f"source_{index}")
        name = _nonempty_string(source["name"], f"source_{index}_name")
        if name not in ALL_SOURCES or name in sources:
            raise ValueError("source_names_must_be_unique_and_frozen")
        role = _nonempty_string(source["role"], f"source_{index}_role")
        if role != ("fallback" if name == ISAAC_GT else "odometry"):
            raise ValueError("source_role_mismatch")
        spec = SourceSpec(
            name=name,
            role=role,
            priority=_integer(source["priority"], f"source_{index}_priority"),
            enabled=_boolean(source["enabled"], f"source_{index}_enabled"),
            odometry_topic=_positive_topic(source["odometry_topic"], "odometry_topic"),
            health_topic=_positive_topic(source["health_topic"], "health_topic"),
            expected_parent_frame=_nonempty_string(
                source["expected_parent_frame"], f"source_{index}_parent_frame"
            ),
            expected_child_frame=_nonempty_string(
                source["expected_child_frame"], f"source_{index}_child_frame"
            ),
            startup_state=_nonempty_string(
                source["startup_state"], f"source_{index}_startup_state"
            ),
            startup_reason=_nonempty_string(
                source["startup_reason"], f"source_{index}_startup_reason"
            ),
        )
        if spec.expected_child_frame != CANONICAL_CHILD_FRAME:
            raise ValueError("source_child_frame_must_be_base_link")
        if spec.startup_state not in {"NO_SAMPLE", "DEFERRED", "UNAVAILABLE"}:
            raise ValueError("invalid_startup_state")
        if not spec.startup_reason:
            raise ValueError("startup_reason_required")
        sources[name] = spec
    if set(sources) != set(ALL_SOURCES):
        raise ValueError("missing_frozen_source")
    priorities = [sources[name].priority for name in (CUVSLAM, LIDAR_IMU)]
    if priorities != sorted(priorities) or len(set(priorities)) != 2:
        raise ValueError("sensor_source_priority_not_frozen")
    if sources[ISAAC_GT].priority <= max(priorities):
        raise ValueError("isaac_gt_must_be_fallback_only")
    frozen_sources = {
        CUVSLAM: {
            "role": "odometry",
            "priority": 10,
            "enabled": True,
            "odometry_topic": "/internvla_t4/localization/source/cuvslam/odometry",
            "health_topic": "/internvla_t4/localization/source/cuvslam/health",
            "expected_parent_frame": "cuvslam_odom",
            "expected_child_frame": "base_link",
            "startup_state": "DEFERRED",
            "startup_reason": "stereo_feed_and_backend_pending_runtime_integration",
        },
        LIDAR_IMU: {
            "role": "odometry",
            "priority": 20,
            "enabled": True,
            "odometry_topic": "/internvla_t4/localization/source/lidar_imu/odometry",
            "health_topic": "/internvla_t4/localization/source/lidar_imu/health",
            "expected_parent_frame": "lidar_imu_odom",
            "expected_child_frame": "base_link",
            "startup_state": "DEFERRED",
            "startup_reason": "lio_backend_and_linear_acceleration_pending_runtime_integration",
        },
        ISAAC_GT: {
            "role": "fallback",
            "priority": 1000,
            "enabled": True,
            "odometry_topic": "/internvla_t4/localization/source/isaac_gt/odometry",
            "health_topic": "/internvla_t4/localization/source/isaac_gt/health",
            "expected_parent_frame": "isaac_world",
            "expected_child_frame": "base_link",
            "startup_state": "NO_SAMPLE",
            "startup_reason": "awaiting_private_isaac_gt_ingress",
        },
    }
    for name, expected in frozen_sources.items():
        actual = {
            field: getattr(sources[name], field) for field in expected
        }
        if actual != expected:
            raise ValueError(f"source_contract_changed:{name}")
    input_topics = {source.odometry_topic for source in sources.values()}

    output = _exact_keys(
        root["output"],
        {
            "odometry_topic",
            "tf_topic",
            "parent_frame",
            "child_frame",
            "source_health_topic",
            "switch_event_topic",
            "deviation_topic",
            "single_authority_required",
            "oracle_completion_topic",
        },
        "output",
    )
    if (
        output["odometry_topic"] != CANONICAL_ODOMETRY_TOPIC
        or output["tf_topic"] != CANONICAL_TF_TOPIC
        or output["parent_frame"] != CANONICAL_PARENT_FRAME
        or output["child_frame"] != CANONICAL_CHILD_FRAME
        or output["single_authority_required"] is not True
        or output["source_health_topic"] != SOURCE_HEALTH_TOPIC
        or output["switch_event_topic"] != SWITCH_EVENT_TOPIC
        or output["deviation_topic"] != DEVIATION_TOPIC
        or output["oracle_completion_topic"]
        != "/internvla_t4/localization/oracle_complete"
    ):
        raise ValueError("canonical_output_contract_changed")
    if output["odometry_topic"] in input_topics:
        raise ValueError("source_input_must_not_equal_selector_output")
    all_topics = [
        *(source.odometry_topic for source in sources.values()),
        *(source.health_topic for source in sources.values()),
        output["odometry_topic"],
        output["source_health_topic"],
        output["switch_event_topic"],
        output["deviation_topic"],
        output["oracle_completion_topic"],
    ]
    if len(all_topics) != len(set(all_topics)):
        raise ValueError("localization_topics_must_be_unique")

    evidence = _exact_keys(
        root["evidence"], {"record_filename", "summary_filename"}, "evidence"
    )
    filenames = (
        _nonempty_string(evidence["record_filename"], "record_filename"),
        _nonempty_string(evidence["summary_filename"], "summary_filename"),
    )
    if filenames != ("localization_records.jsonl", "localization_summary.json"):
        raise ValueError("evidence_filenames_changed")
    if any("/" in name or "\\" in name or not name for name in filenames):
        raise ValueError("evidence_filenames_must_be_leaf_names")
    return LocalizationConfig(
        schema_version=1,
        policy=policy,
        allowed_targets=allowed,
        forbidden_targets=forbidden,
        use_sim_time=True,
        sources=sources,
        output_health_topic=_positive_topic(output["source_health_topic"], "health_topic"),
        output_switch_topic=_positive_topic(output["switch_event_topic"], "switch_topic"),
        output_deviation_topic=_positive_topic(output["deviation_topic"], "deviation_topic"),
        oracle_completion_topic=_positive_topic(
            output["oracle_completion_topic"], "oracle_completion_topic"
        ),
        evidence_record_filename=filenames[0],
        evidence_summary_filename=filenames[1],
    )
