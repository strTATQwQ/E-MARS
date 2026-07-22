"""Fail-closed, deterministic contract for the frozen T4.6 ablation matrix.

This module is deliberately standard-library-only and offline.  It does not import
ROS, Isaac, model runners, subprocess, sockets, or resource-lease code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """Raised when an ablation artifact violates the frozen contract."""


MATRIX_ID = "internnav_t4_6_completion_sim_ablation_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROFILES = {
    "completion_sim": {
        "path": "configs/runtime/completion_sim.yaml",
        "sha256": "5219f2d9d6496dcff889f563677e60c666aed322b07776dd7da61295b0d090cc",
    },
    "strict_evidence": {
        "path": "configs/runtime/strict_evidence.yaml",
        "sha256": "96e3c3fba54ff14c3ce245c2acaca9aa2848c074d00d1d8b99f661f262e5ba50",
    },
}
VARIANT_IDS = (
    "full_system1_system2",
    "oracle_high_level_system1",
    "system2_oracle_local_path",
    "full_trajectory",
    "endpoint",
    "straight_line",
    "model_stop",
    "oracle_termination",
    "history_on",
    "history_off",
    "recovery_on",
    "recovery_off",
    "h1_view",
    "go2_view",
)
COMPARISON_IDS = (
    "system2_gap_to_oracle_high_level",
    "system1_gap_to_oracle_local_path",
    "full_trajectory_vs_endpoint",
    "full_trajectory_vs_straight_line",
    "model_stop_gap_to_oracle_termination",
    "history_on_vs_off",
    "recovery_on_vs_off",
    "go2_view_vs_h1_view",
)
FACTOR_LEVELS = {
    "system_mode": [
        "full_system1_system2",
        "oracle_high_level_system1",
        "system2_oracle_local_path",
    ],
    "trajectory_mode": ["full_trajectory", "endpoint", "straight_line"],
    "termination_mode": ["model_stop", "oracle_termination"],
    "history_mode": ["on", "off"],
    "recovery_mode": ["on", "off"],
    "view_mode": ["go2_view", "h1_view"],
}
BASELINE_FACTORS = {
    "system_mode": "full_system1_system2",
    "trajectory_mode": "full_trajectory",
    "termination_mode": "model_stop",
    "history_mode": "on",
    "recovery_mode": "off",
    "view_mode": "go2_view",
}
CONTROL_REPEAT_VARIANTS = (
    "full_system1_system2",
    "full_trajectory",
    "model_stop",
    "history_on",
    "recovery_off",
    "go2_view",
)
WARNING_ALLOWLIST = (
    "recorder_partial_frame",
    "tf_nearest_or_latest",
    "source_timeout_extended",
    "watchdog_extended",
    "costmap_current_extended",
    "collision_monitor_warn_only",
    "nvblox_shadow",
    "render_stamp_deviation",
)
ORACLE_UNSTABLE_REASONS = (
    "stop_unstable_missing",
    "stop_unstable_premature",
    "stop_unstable_oscillatory",
)
METRICS = (
    ("sr", "official_metrics.sr", "fraction", "higher"),
    ("os", "official_metrics.os", "fraction", "higher"),
    ("spl", "official_metrics.spl", "fraction", "higher"),
    ("ne_m", "official_metrics.ne_m", "m", "lower"),
    ("stuck", "diagnostics.stuck", "fraction", "lower"),
    (
        "latency_system1_mean_ms",
        "diagnostics.latency_ms.system1.mean",
        "ms",
        "lower",
    ),
    (
        "latency_system1_p95_ms",
        "diagnostics.latency_ms.system1.p95",
        "ms",
        "lower",
    ),
    (
        "latency_system2_mean_ms",
        "diagnostics.latency_ms.system2.mean",
        "ms",
        "lower",
    ),
    (
        "latency_system2_p95_ms",
        "diagnostics.latency_ms.system2.p95",
        "ms",
        "lower",
    ),
    (
        "latency_end_to_end_mean_ms",
        "diagnostics.latency_ms.end_to_end.mean",
        "ms",
        "lower",
    ),
    (
        "latency_end_to_end_p95_ms",
        "diagnostics.latency_ms.end_to_end.p95",
        "ms",
        "lower",
    ),
)

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PATH_RE = re.compile(r"^(?!/)(?![A-Za-z]:)(?!.*(?:^|/)\.\.(?:/|$))[^\\]+$")


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"invalid {label}: {exc}") from exc


def load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read {path.as_posix()}: {exc}") from exc
    return strict_json_loads(text, label=path.as_posix())


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_text_file_sha256(path: Path) -> str:
    """Hash reviewed text independently of Git checkout line endings."""

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read frozen runtime profile {path}: {exc}") from exc
    normalized = raw.replace(b"\r\n", b"\n")
    if b"\r" in normalized:
        raise ContractError(f"frozen runtime profile contains a lone carriage return: {path}")
    try:
        normalized.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"frozen runtime profile is not UTF-8 text: {path}") from exc
    return hashlib.sha256(normalized).hexdigest()


def json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return canonical_json_bytes(left) == canonical_json_bytes(right)


def pretty_json(value: Any) -> str:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not writable JSON: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.write_text(pretty_json(value), encoding="utf-8", newline="\n")


def expect_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{label} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def expect_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    return value


def expect_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError(f"{label} must be a finite representable number") from exc
    if not math.isfinite(converted):
        raise ContractError(f"{label} must be finite")
    if minimum is not None and converted < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    if maximum is not None and converted > maximum:
        raise ContractError(f"{label} must be <= {maximum}")
    return converted


def expect_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        raise ContractError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def expect_safe_path(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _SAFE_PATH_RE.fullmatch(value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ContractError(f"{label} must be a safe repository-relative path")
    return value


def _factors_with(**changes: str) -> dict[str, str]:
    result = dict(BASELINE_FACTORS)
    result.update(changes)
    return result


def _expected_variants() -> list[dict[str, Any]]:
    system_comparisons = [COMPARISON_IDS[0], COMPARISON_IDS[1]]
    return [
        {
            "variant_id": "full_system1_system2",
            "comparison_block": "system",
            "role": "candidate",
            "comparison_ids": system_comparisons,
            "factors": _factors_with(),
        },
        {
            "variant_id": "oracle_high_level_system1",
            "comparison_block": "system",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[0]],
            "factors": _factors_with(system_mode="oracle_high_level_system1"),
        },
        {
            "variant_id": "system2_oracle_local_path",
            "comparison_block": "system",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[1]],
            "factors": _factors_with(system_mode="system2_oracle_local_path"),
        },
        {
            "variant_id": "full_trajectory",
            "comparison_block": "trajectory",
            "role": "candidate",
            "comparison_ids": [COMPARISON_IDS[2], COMPARISON_IDS[3]],
            "factors": _factors_with(),
        },
        {
            "variant_id": "endpoint",
            "comparison_block": "trajectory",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[2]],
            "factors": _factors_with(trajectory_mode="endpoint"),
        },
        {
            "variant_id": "straight_line",
            "comparison_block": "trajectory",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[3]],
            "factors": _factors_with(trajectory_mode="straight_line"),
        },
        {
            "variant_id": "model_stop",
            "comparison_block": "stop",
            "role": "candidate",
            "comparison_ids": [COMPARISON_IDS[4]],
            "factors": _factors_with(),
        },
        {
            "variant_id": "oracle_termination",
            "comparison_block": "stop",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[4]],
            "factors": _factors_with(termination_mode="oracle_termination"),
        },
        {
            "variant_id": "history_on",
            "comparison_block": "history",
            "role": "candidate",
            "comparison_ids": [COMPARISON_IDS[5]],
            "factors": _factors_with(),
        },
        {
            "variant_id": "history_off",
            "comparison_block": "history",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[5]],
            "factors": _factors_with(history_mode="off"),
        },
        {
            "variant_id": "recovery_on",
            "comparison_block": "recovery",
            "role": "candidate",
            "comparison_ids": [COMPARISON_IDS[6]],
            "factors": _factors_with(recovery_mode="on"),
        },
        {
            "variant_id": "recovery_off",
            "comparison_block": "recovery",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[6]],
            "factors": _factors_with(),
        },
        {
            "variant_id": "h1_view",
            "comparison_block": "view",
            "role": "reference",
            "comparison_ids": [COMPARISON_IDS[7]],
            "factors": _factors_with(view_mode="h1_view"),
        },
        {
            "variant_id": "go2_view",
            "comparison_block": "view",
            "role": "candidate",
            "comparison_ids": [COMPARISON_IDS[7]],
            "factors": _factors_with(),
        },
    ]


def _expected_comparisons() -> list[dict[str, Any]]:
    return [
        {
            "comparison_id": COMPARISON_IDS[0],
            "comparison_block": "system",
            "component": "system2",
            "candidate_variant": "full_system1_system2",
            "reference_variant": "oracle_high_level_system1",
            "axis": "system_mode",
            "effect_kind": "replacement_gap",
            "oracle_assisted": True,
            "interpretation": "model System2 relative to an oracle high-level replacement; not a pure on/off causal effect",
        },
        {
            "comparison_id": COMPARISON_IDS[1],
            "comparison_block": "system",
            "component": "system1",
            "candidate_variant": "full_system1_system2",
            "reference_variant": "system2_oracle_local_path",
            "axis": "system_mode",
            "effect_kind": "replacement_gap",
            "oracle_assisted": True,
            "interpretation": "model System1 trajectory relative to an oracle local-path replacement; not a pure on/off causal effect",
        },
        {
            "comparison_id": COMPARISON_IDS[2],
            "comparison_block": "trajectory",
            "component": "trajectory_endpoint",
            "candidate_variant": "full_trajectory",
            "reference_variant": "endpoint",
            "axis": "trajectory_mode",
            "effect_kind": "paired_contribution",
            "oracle_assisted": False,
            "interpretation": "full trajectory relative to endpoint-only output",
        },
        {
            "comparison_id": COMPARISON_IDS[3],
            "comparison_block": "trajectory",
            "component": "trajectory_straight_line",
            "candidate_variant": "full_trajectory",
            "reference_variant": "straight_line",
            "axis": "trajectory_mode",
            "effect_kind": "paired_contribution",
            "oracle_assisted": False,
            "interpretation": "full trajectory relative to a straight-line transform",
        },
        {
            "comparison_id": COMPARISON_IDS[4],
            "comparison_block": "stop",
            "component": "stop",
            "candidate_variant": "model_stop",
            "reference_variant": "oracle_termination",
            "axis": "termination_mode",
            "effect_kind": "gap_to_oracle",
            "oracle_assisted": True,
            "interpretation": "model STOP gap to oracle termination; diagnostic upper bound, never model credit",
        },
        {
            "comparison_id": COMPARISON_IDS[5],
            "comparison_block": "history",
            "component": "history",
            "candidate_variant": "history_on",
            "reference_variant": "history_off",
            "axis": "history_mode",
            "effect_kind": "paired_contribution",
            "oracle_assisted": False,
            "interpretation": "history enabled relative to history disabled",
        },
        {
            "comparison_id": COMPARISON_IDS[6],
            "comparison_block": "recovery",
            "component": "recovery",
            "candidate_variant": "recovery_on",
            "reference_variant": "recovery_off",
            "axis": "recovery_mode",
            "effect_kind": "paired_contribution",
            "oracle_assisted": False,
            "interpretation": "recovery enabled relative to recovery disabled; requires observed recovery triggers",
        },
        {
            "comparison_id": COMPARISON_IDS[7],
            "comparison_block": "view",
            "component": "view",
            "candidate_variant": "go2_view",
            "reference_variant": "h1_view",
            "axis": "view_mode",
            "effect_kind": "paired_contribution",
            "oracle_assisted": False,
            "interpretation": "Go2 visual input relative to H1 visual input; RGB-only with mapping and safety geometry frozen",
        },
    ]


def _expected_metrics() -> list[dict[str, str]]:
    return [
        {
            "metric_id": metric_id,
            "path": path,
            "unit": unit,
            "benefit_direction": direction,
        }
        for metric_id, path, unit, direction in METRICS
    ]


def validate_matrix(matrix: Any) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "matrix_id",
        "status",
        "runtime_policy",
        "runtime_profiles",
        "target",
        "allowed_evidence_kinds",
        "forbidden_targets",
        "factor_levels",
        "baseline_factors",
        "control_repeat_variants",
        "variants",
        "comparisons",
        "metrics",
        "pairing",
        "statistics",
        "oracle_termination_contract",
        "completion_sim_invariants",
        "schemas",
    }
    value = expect_exact_keys(matrix, expected_keys, "matrix")
    if expect_int(value["schema_version"], "matrix.schema_version") != 1:
        raise ContractError("matrix.schema_version must equal 1")
    frozen_scalars = {
        "matrix_id": MATRIX_ID,
        "status": "FROZEN",
        "runtime_policy": "completion_sim",
        "target": "isaac_simulation_only",
    }
    for key, expected in frozen_scalars.items():
        if value[key] != expected:
            raise ContractError(f"matrix.{key} must equal {expected!r}")
    if not json_values_equal(
        value["allowed_evidence_kinds"], ["completion_sim", "synthetic"]
    ):
        raise ContractError("matrix.allowed_evidence_kinds is not frozen")
    if not json_values_equal(value["forbidden_targets"], ["real_go2", "hardware_motion"]):
        raise ContractError("matrix.forbidden_targets is not frozen")
    if not json_values_equal(value["runtime_profiles"], RUNTIME_PROFILES):
        raise ContractError("matrix.runtime_profiles is not frozen")
    for profile_name, profile in RUNTIME_PROFILES.items():
        profile_path = REPOSITORY_ROOT / profile["path"]
        actual_sha256 = canonical_text_file_sha256(profile_path)
        if actual_sha256 != profile["sha256"]:
            raise ContractError(f"runtime profile content changed: {profile_name}")
    if not json_values_equal(value["factor_levels"], FACTOR_LEVELS):
        raise ContractError("matrix.factor_levels is not frozen")
    if not json_values_equal(value["baseline_factors"], BASELINE_FACTORS):
        raise ContractError("matrix.baseline_factors is not frozen")
    if not json_values_equal(
        value["control_repeat_variants"], list(CONTROL_REPEAT_VARIANTS)
    ):
        raise ContractError("matrix.control_repeat_variants is not frozen")
    if not json_values_equal(value["variants"], _expected_variants()):
        raise ContractError("matrix.variants is not the frozen 14-arm matrix")
    if not json_values_equal(value["comparisons"], _expected_comparisons()):
        raise ContractError("matrix.comparisons is not the frozen 8-contrast set")
    if not json_values_equal(value["metrics"], _expected_metrics()):
        raise ContractError("matrix.metrics is not frozen")

    pairing = expect_exact_keys(
        value["pairing"],
        {
            "minimum_episode_count",
            "same_dataset_sha256_required",
            "same_episode_manifest_sha256_required",
            "same_episode_order_required",
            "pairing_keys",
        },
        "matrix.pairing",
    )
    if not json_values_equal(pairing, {
        "minimum_episode_count": 20,
        "same_dataset_sha256_required": True,
        "same_episode_manifest_sha256_required": True,
        "same_episode_order_required": True,
        "pairing_keys": [
            "dataset_sha256",
            "episode_id",
            "episode_ordinal",
            "seed",
        ],
    }):
        raise ContractError("matrix.pairing is not frozen")

    statistics = expect_exact_keys(
        value["statistics"],
        {"bootstrap_replicates", "confidence_level", "seed_namespace"},
        "matrix.statistics",
    )
    if not json_values_equal(statistics, {
        "bootstrap_replicates": 2000,
        "confidence_level": 0.95,
        "seed_namespace": "internnav-t4.6-ablation-v1",
    }):
        raise ContractError("matrix.statistics is not frozen")

    oracle = expect_exact_keys(
        value["oracle_termination_contract"],
        {
            "scope",
            "unstable_reasons",
            "threshold_m",
            "threshold_source",
            "distance_source",
            "requires_original_model_stop_observation",
            "requires_episode_identity_match",
            "interpretation",
        },
        "matrix.oracle_termination_contract",
    )
    if not json_values_equal(oracle, {
        "scope": "termination_only",
        "unstable_reasons": list(ORACLE_UNSTABLE_REASONS),
        "threshold_m": 2.5,
        "threshold_source": "frozen_evaluator_success_radius",
        "distance_source": "evaluator_ground_truth",
        "requires_original_model_stop_observation": True,
        "requires_episode_identity_match": True,
        "interpretation": "diagnostic_upper_bound",
    }):
        raise ContractError("matrix.oracle_termination_contract is not frozen")

    invariants = expect_exact_keys(
        value["completion_sim_invariants"],
        {
            "strict_evidence_unchanged",
            "bounded_velocity_required",
            "simulation_estop_required",
            "view_scope",
            "mapping_and_safety_geometry",
            "warning_allowlist",
            "forbid_step_3_7",
            "forbid_nvfp4",
            "forbid_model_finetuning",
            "forbid_model_execution_in_worker",
        },
        "matrix.completion_sim_invariants",
    )
    expected_invariants = {
        "strict_evidence_unchanged": True,
        "bounded_velocity_required": True,
        "simulation_estop_required": True,
        "view_scope": "internvla_rgb_only",
        "mapping_and_safety_geometry": "go2_frozen",
        "warning_allowlist": list(WARNING_ALLOWLIST),
        "forbid_step_3_7": True,
        "forbid_nvfp4": True,
        "forbid_model_finetuning": True,
        "forbid_model_execution_in_worker": True,
    }
    if not json_values_equal(invariants, expected_invariants):
        raise ContractError("matrix.completion_sim_invariants is not frozen")

    schemas = expect_exact_keys(
        value["schemas"],
        {"variant_config", "episode_record", "episode_level", "aggregate"},
        "matrix.schemas",
    )
    expected_schemas = {
        "variant_config": "configs/completion_sim/ablation/variant_config_schema_v1.json",
        "episode_record": "configs/completion_sim/ablation/episode_record_schema_v1.json",
        "episode_level": "configs/completion_sim/ablation/episode_level_schema_v1.json",
        "aggregate": "configs/completion_sim/ablation/aggregate_schema_v1.json",
    }
    if not json_values_equal(schemas, expected_schemas):
        raise ContractError("matrix.schemas is not frozen")
    for key, path in schemas.items():
        expect_safe_path(path, f"matrix.schemas.{key}")

    # Redundant checks make the mutual-exclusion invariant explicit and local.
    for variant in value["variants"]:
        factors = variant["factors"]
        expect_exact_keys(factors, set(FACTOR_LEVELS), f"{variant['variant_id']}.factors")
        for factor, levels in FACTOR_LEVELS.items():
            if factors[factor] not in levels:
                raise ContractError(
                    f"{variant['variant_id']}.{factor} is not exactly one frozen level"
                )
    variants_by_id = {item["variant_id"]: item for item in value["variants"]}
    for comparison in value["comparisons"]:
        candidate = variants_by_id[comparison["candidate_variant"]]["factors"]
        reference = variants_by_id[comparison["reference_variant"]]["factors"]
        changed = [name for name in FACTOR_LEVELS if candidate[name] != reference[name]]
        if changed != [comparison["axis"]]:
            raise ContractError(
                f"{comparison['comparison_id']} changes {changed}, expected only "
                f"{comparison['axis']}"
            )
    return value


def load_matrix(path: Path) -> dict[str, Any]:
    return validate_matrix(load_json(path))


def resolve_variant_configs(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_matrix(matrix)
    matrix_sha256 = canonical_sha256(matrix)
    invariants = matrix["completion_sim_invariants"]
    oracle_contract = matrix["oracle_termination_contract"]
    result: dict[str, dict[str, Any]] = {}
    for variant in matrix["variants"]:
        factors = deepcopy(variant["factors"])
        result[variant["variant_id"]] = {
            "schema_version": 1,
            "artifact_type": "t4_ablation_variant_config",
            "matrix_id": matrix["matrix_id"],
            "matrix_sha256": matrix_sha256,
            "variant_id": variant["variant_id"],
            "comparison_block": variant["comparison_block"],
            "comparison_ids": list(variant["comparison_ids"]),
            "runtime_policy": matrix["runtime_policy"],
            "runtime_profiles": deepcopy(matrix["runtime_profiles"]),
            "target": matrix["target"],
            "factors": factors,
            "view_scope": invariants["view_scope"],
            "mapping_and_safety_geometry": invariants[
                "mapping_and_safety_geometry"
            ],
            "strict_evidence_unchanged": invariants["strict_evidence_unchanged"],
            "bounded_velocity_required": invariants["bounded_velocity_required"],
            "simulation_estop_required": invariants["simulation_estop_required"],
            "forbid_step_3_7": invariants["forbid_step_3_7"],
            "forbid_nvfp4": invariants["forbid_nvfp4"],
            "forbid_model_finetuning": invariants["forbid_model_finetuning"],
            "forbid_model_execution_in_worker": invariants[
                "forbid_model_execution_in_worker"
            ],
            "forbidden_targets": list(matrix["forbidden_targets"]),
            "warning_allowlist": list(invariants["warning_allowlist"]),
            "oracle_termination": {
                "enabled": factors["termination_mode"] == "oracle_termination",
                "scope": oracle_contract["scope"],
                "unstable_reasons": list(oracle_contract["unstable_reasons"]),
                "threshold_m": oracle_contract["threshold_m"],
                "threshold_source": oracle_contract["threshold_source"],
                "distance_source": oracle_contract["distance_source"],
                "requires_original_model_stop_observation": oracle_contract[
                    "requires_original_model_stop_observation"
                ],
                "requires_episode_identity_match": oracle_contract[
                    "requires_episode_identity_match"
                ],
                "interpretation": oracle_contract["interpretation"],
            },
        }
    return result


def variant_config_hashes(matrix: dict[str, Any]) -> dict[str, str]:
    return {
        variant_id: canonical_sha256(config)
        for variant_id, config in resolve_variant_configs(matrix).items()
    }


def variant_config_manifest(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_matrix(matrix)
    configs = resolve_variant_configs(matrix)
    return {
        "schema_version": 1,
        "artifact_type": "t4_ablation_variant_manifest",
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": canonical_sha256(matrix),
        "variant_order": list(VARIANT_IDS),
        "variants": [
            {
                "variant_id": variant_id,
                "path": f"variants/{variant_id}.json",
                "config_sha256": canonical_sha256(configs[variant_id]),
                "factor_signature_sha256": canonical_sha256(
                    configs[variant_id]["factors"]
                ),
            }
            for variant_id in VARIANT_IDS
        ],
        "schemas": deepcopy(matrix["schemas"]),
        "resource_use": "none",
        "online_execution": False,
    }


def validate_variant_config(
    matrix: dict[str, Any], variant_id: str, config: Any
) -> dict[str, Any]:
    configs = resolve_variant_configs(matrix)
    if variant_id not in configs:
        raise ContractError(f"unknown variant_id: {variant_id}")
    if not json_values_equal(config, configs[variant_id]):
        raise ContractError(f"variant config does not match frozen arm: {variant_id}")
    return config


def matrix_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    validate_matrix(matrix)
    configs = resolve_variant_configs(matrix)
    return {
        "status": "VALID",
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": canonical_sha256(matrix),
        "variant_count": len(configs),
        "comparison_count": len(matrix["comparisons"]),
        "minimum_episode_count": matrix["pairing"]["minimum_episode_count"],
        "resource_use": "none",
    }
