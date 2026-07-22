"""Deterministic paired contribution statistics for T4.6 episode records."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from typing import Any, Iterable

from .contract import (
    ContractError,
    CONTROL_REPEAT_VARIANTS,
    VARIANT_IDS,
    canonical_sha256,
    json_values_equal,
    resolve_variant_configs,
    validate_matrix,
    variant_config_hashes,
)
from .records import ValidatedCollection, pairing_key, validate_episode_collection


_ACTIVATION_COUNT_KEYS = (
    "system1_count",
    "system2_count",
    "oracle_high_level_count",
    "oracle_local_path_count",
    "trajectory_transform_count",
    "history_frame_count",
    "history_reset_count",
    "recovery_trigger_count",
    "visual_frame_count",
)
_SAFETY_COUNT_KEYS = (
    "collision_count",
    "fall_count",
    "stale_command_count",
    "reset_contamination_count",
    "bounded_velocity_violation_count",
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ContractError("cannot compute the mean of an empty sample")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    try:
        result = scale * (math.fsum(value / scale for value in values) / len(values))
    except OverflowError as exc:
        raise ContractError("finite metric values overflowed stable mean") from exc
    if not math.isfinite(result):
        raise ContractError("finite metric values produced a non-finite mean")
    return result


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ContractError("cannot compute a quantile of an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("quantile probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    result = (1.0 - fraction) * sorted_values[lower] + fraction * sorted_values[upper]
    if not math.isfinite(result):
        raise ContractError("finite bootstrap values produced a non-finite quantile")
    return result


def _median(values: list[float]) -> float:
    if not values:
        raise ContractError("cannot compute the median of an empty sample")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return _mean([ordered[midpoint - 1], ordered[midpoint]])


def paired_bootstrap_ci(
    deltas: Iterable[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> list[float]:
    values = [float(value) for value in deltas]
    if not values or any(not math.isfinite(value) for value in values):
        raise ContractError("paired bootstrap requires finite, nonempty deltas")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ContractError("bootstrap replicates must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ContractError("bootstrap confidence_level must be in (0, 1)")
    if len(set(values)) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    count = len(values)
    bootstrap_means = [
        _mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(replicates)
    ]
    bootstrap_means.sort()
    tail = (1.0 - confidence_level) / 2.0
    return [
        _quantile(bootstrap_means, tail),
        _quantile(bootstrap_means, 1.0 - tail),
    ]


def _bootstrap_seed(namespace: str, comparison_id: str, metric_id: str) -> int:
    digest = hashlib.sha256(
        f"{namespace}\0{comparison_id}\0{metric_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _nested_value(record: dict[str, Any], path: str) -> float | None:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"metric path is missing from episode record: {path}")
        value = value[part]
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"metric path is not finite numeric or null: {path}")
    return float(value)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "min": min(values),
        "max": max(values),
    }


def _variant_summary(
    matrix: dict[str, Any],
    variant_id: str,
    rows: tuple[dict[str, Any], ...],
    config_hashes: dict[str, str],
) -> dict[str, Any]:
    variant = next(item for item in matrix["variants"] if item["variant_id"] == variant_id)
    metrics: dict[str, Any] = {}
    for definition in matrix["metrics"]:
        values = [
            value
            for row in rows
            if (value := _nested_value(row, definition["path"])) is not None
        ]
        metrics[definition["metric_id"]] = _summary(values)

    termination_reasons = Counter(row["termination"]["reason"] for row in rows)
    effective_sources = Counter(
        row["termination"]["effective_source"] for row in rows
    )
    activation_counts = {
        name: sum(row["activation"][name] for row in rows)
        for name in _ACTIVATION_COUNT_KEYS
    }
    safety_counts = {
        name: sum(row["safety"][name] for row in rows)
        for name in _SAFETY_COUNT_KEYS
    }
    warnings = Counter(warning for row in rows for warning in row["warnings"])
    return {
        "variant_id": variant_id,
        "factors": variant["factors"],
        "factor_signature_sha256": canonical_sha256(variant["factors"]),
        "variant_config_sha256": config_hashes[variant_id],
        "episode_count": len(rows),
        "metrics": metrics,
        "termination": {
            "reason_counts": dict(sorted(termination_reasons.items())),
            "effective_source_counts": dict(sorted(effective_sources.items())),
            "oracle_mode_episode_count": sum(
                row["termination"]["oracle_mode_enabled"] for row in rows
            ),
            "oracle_terminated_episode_count": sum(
                row["termination"]["terminated_by_oracle"] for row in rows
            ),
            "model_stop_observed_episode_count": sum(
                row["termination"]["model_stop_observed"] for row in rows
            ),
            "model_stop_suppressed_count": sum(
                row["termination"]["model_stop_suppressed_count"] for row in rows
            ),
        },
        "activation_counts": activation_counts,
        "safety_counts": safety_counts,
        "warning_counts": dict(sorted(warnings.items())),
    }


def _component_activation(record: dict[str, Any], component: str, *, candidate: bool) -> int:
    activation = record["activation"]
    if component == "system1":
        return activation["system1_count"] if candidate else activation["oracle_local_path_count"]
    if component == "system2":
        return activation["system2_count"] if candidate else activation["oracle_high_level_count"]
    if component.startswith("trajectory_"):
        return activation["trajectory_transform_count"]
    if component == "stop":
        if candidate:
            return int(record["termination"]["model_stop_observed"])
        return int(record["termination"]["terminated_by_oracle"])
    if component == "history":
        return activation["history_frame_count"]
    if component == "recovery":
        return activation["recovery_trigger_count"]
    if component == "view":
        return activation["visual_frame_count"]
    raise ContractError(f"unknown comparison component: {component}")


def _paired_metric_summary(
    metric_definition: dict[str, str],
    episode_metric_rows: list[dict[str, Any]],
    *,
    statistics_config: dict[str, Any],
    comparison_id: str,
    comparison_estimation_status: str,
) -> dict[str, Any]:
    raw = [
        row["raw_candidate_minus_reference"]
        for row in episode_metric_rows
        if row["raw_candidate_minus_reference"] is not None
    ]
    benefit = [
        row["benefit_oriented_delta"]
        for row in episode_metric_rows
        if row["benefit_oriented_delta"] is not None
    ]
    if len(raw) != len(benefit):
        raise ContractError("raw and oriented paired metric counts disagree")
    total = len(episode_metric_rows)
    if comparison_estimation_status != "ESTIMATED":
        status = comparison_estimation_status
        paired_value_count = 0
        raw_mean = None
        benefit_mean = None
        benefit_median = None
        ci95 = None
        wins = ties = losses = 0
    elif not benefit:
        status = "NOT_ESTIMABLE_MISSING_VALUES"
        paired_value_count = 0
        raw_mean = None
        benefit_mean = None
        benefit_median = None
        ci95 = None
        wins = ties = losses = 0
    else:
        status = "ESTIMATED" if len(benefit) == total else "PARTIAL_MISSING_VALUES"
        paired_value_count = len(benefit)
        raw_mean = _mean(raw)
        benefit_mean = _mean(benefit)
        benefit_median = _median(benefit)
        ci95 = paired_bootstrap_ci(
            benefit,
            replicates=statistics_config["bootstrap_replicates"],
            confidence_level=statistics_config["confidence_level"],
            seed=_bootstrap_seed(
                statistics_config["seed_namespace"],
                comparison_id,
                metric_definition["metric_id"],
            ),
        )
        wins = sum(value > 0.0 for value in benefit)
        ties = sum(value == 0.0 for value in benefit)
        losses = sum(value < 0.0 for value in benefit)
    return {
        "unit": metric_definition["unit"],
        "benefit_direction": metric_definition["benefit_direction"],
        "total_pair_count": total,
        "paired_value_count": paired_value_count,
        "estimation_status": status,
        "raw_mean_delta": raw_mean,
        "benefit_mean_delta": benefit_mean,
        "benefit_median_delta": benefit_median,
        "benefit_ci95": ci95,
        "candidate_wins": wins,
        "ties": ties,
        "reference_wins": losses,
    }


def _control_drift(
    matrix: dict[str, Any], variant_summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    factor_hashes = {
        variant_summaries[variant_id]["factor_signature_sha256"]
        for variant_id in CONTROL_REPEAT_VARIANTS
    }
    if len(factor_hashes) != 1:
        raise ContractError("control repeat variants do not share one factor signature")
    metrics: dict[str, Any] = {}
    for definition in matrix["metrics"]:
        metric_id = definition["metric_id"]
        means = [
            variant_summaries[variant_id]["metrics"][metric_id]["mean"]
            for variant_id in CONTROL_REPEAT_VARIANTS
        ]
        present = [float(value) for value in means if value is not None]
        if len(present) != len(means):
            metrics[metric_id] = {
                "control_count": len(means),
                "observed_count": len(present),
                "minimum_mean": None,
                "maximum_mean": None,
                "span": None,
            }
        else:
            metrics[metric_id] = {
                "control_count": len(means),
                "observed_count": len(present),
                "minimum_mean": min(present),
                "maximum_mean": max(present),
                "span": max(present) - min(present),
            }
    return {
        "control_variants": list(CONTROL_REPEAT_VARIANTS),
        "factor_signature_sha256": next(iter(factor_hashes)),
        "metrics": metrics,
    }


def build_analysis_outputs(
    matrix: dict[str, Any], records: list[dict[str, Any]] | tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_matrix(matrix)
    collection: ValidatedCollection = validate_episode_collection(records, matrix)
    matrix_sha = canonical_sha256(matrix)
    config_hashes = variant_config_hashes(matrix)
    variant_summaries = {
        variant_id: _variant_summary(
            matrix, variant_id, collection.by_variant[variant_id], config_hashes
        )
        for variant_id in VARIANT_IDS
    }
    paired_contributions: list[dict[str, Any]] = []
    comparison_summaries: list[dict[str, Any]] = []
    metric_definitions = {item["metric_id"]: item for item in matrix["metrics"]}

    for comparison in matrix["comparisons"]:
        candidate_rows = {
            pairing_key(row): row
            for row in collection.by_variant[comparison["candidate_variant"]]
        }
        reference_rows = {
            pairing_key(row): row
            for row in collection.by_variant[comparison["reference_variant"]]
        }
        if set(candidate_rows) != set(reference_rows):
            raise ContractError(f"pairing changed during {comparison['comparison_id']}")
        episode_metric_rows_by_id: dict[str, list[dict[str, Any]]] = {
            metric_id: [] for metric_id in metric_definitions
        }
        oracle_assisted_episode_count = 0
        recovery_triggered_pair_count = 0
        candidate_activation_count = 0
        reference_activation_count = 0
        for key in sorted(candidate_rows, key=lambda item: item[2]):
            candidate = candidate_rows[key]
            reference = reference_rows[key]
            metrics: dict[str, Any] = {}
            for metric_id, definition in metric_definitions.items():
                candidate_value = _nested_value(candidate, definition["path"])
                reference_value = _nested_value(reference, definition["path"])
                if candidate_value is None or reference_value is None:
                    raw_delta = None
                    benefit_delta = None
                else:
                    raw_delta = candidate_value - reference_value
                    direction = 1.0 if definition["benefit_direction"] == "higher" else -1.0
                    benefit_delta = direction * raw_delta
                metric_row = {
                    "candidate_value": candidate_value,
                    "reference_value": reference_value,
                    "raw_candidate_minus_reference": raw_delta,
                    "benefit_oriented_delta": benefit_delta,
                }
                metrics[metric_id] = metric_row
                episode_metric_rows_by_id[metric_id].append(metric_row)

            oracle_assisted_episode = (
                reference["termination"]["terminated_by_oracle"]
                if comparison["comparison_block"] == "stop"
                else comparison["comparison_block"] == "system"
            )
            oracle_assisted_episode_count += int(oracle_assisted_episode)
            recovery_triggered_pair = (
                candidate["activation"]["recovery_trigger_count"] > 0
                if comparison["comparison_block"] == "recovery"
                else False
            )
            recovery_triggered_pair_count += int(recovery_triggered_pair)
            candidate_activation_count += _component_activation(
                candidate, comparison["component"], candidate=True
            )
            reference_activation_count += _component_activation(
                reference, comparison["component"], candidate=False
            )
            paired_contributions.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "component": comparison["component"],
                    "effect_kind": comparison["effect_kind"],
                    "interpretation": comparison["interpretation"],
                    "candidate_variant": comparison["candidate_variant"],
                    "reference_variant": comparison["reference_variant"],
                    "episode_id": candidate["episode_id"],
                    "episode_ordinal": candidate["episode_ordinal"],
                    "seed": candidate["seed"],
                    "oracle_assisted_episode": oracle_assisted_episode,
                    "recovery_triggered_pair": recovery_triggered_pair,
                    "metrics": metrics,
                }
            )

        if comparison["comparison_block"] == "stop" and oracle_assisted_episode_count == 0:
            estimation_status = "NOT_ESTIMABLE_NO_ORACLE_TERMINATION"
        elif (
            comparison["comparison_block"] == "recovery"
            and recovery_triggered_pair_count == 0
        ):
            estimation_status = "NOT_ESTIMABLE_NO_RECOVERY_TRIGGER"
        else:
            estimation_status = "ESTIMATED"
        metric_summaries = {
            metric_id: _paired_metric_summary(
                definition,
                episode_metric_rows_by_id[metric_id],
                statistics_config=matrix["statistics"],
                comparison_id=comparison["comparison_id"],
                comparison_estimation_status=estimation_status,
            )
            for metric_id, definition in metric_definitions.items()
        }
        comparison_summaries.append(
            {
                "comparison_id": comparison["comparison_id"],
                "comparison_block": comparison["comparison_block"],
                "component": comparison["component"],
                "candidate_variant": comparison["candidate_variant"],
                "reference_variant": comparison["reference_variant"],
                "axis": comparison["axis"],
                "effect_kind": comparison["effect_kind"],
                "interpretation": comparison["interpretation"],
                "paired_episode_count": collection.episode_count_per_variant,
                "estimation_status": estimation_status,
                "oracle_assisted": comparison["oracle_assisted"],
                "oracle_assisted_episode_count": oracle_assisted_episode_count,
                "recovery_triggered_pair_count": recovery_triggered_pair_count,
                "candidate_activation_count": candidate_activation_count,
                "reference_activation_count": reference_activation_count,
                "metrics": metric_summaries,
            }
        )

    analysis_status = (
        "SYNTHETIC_COMPLETE"
        if collection.evidence_kind == "synthetic"
        else "COMPLETE"
    )
    shared = {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": matrix_sha,
        "analysis_status": analysis_status,
        "evidence_kind": collection.evidence_kind,
        "runtime_policy": matrix["runtime_policy"],
        "execution_target": collection.execution_target,
        "runner_commit_sha": collection.runner_commit_sha,
        "dataset_sha256": collection.dataset_sha256,
        "episode_manifest_sha256": collection.episode_manifest_sha256,
        "episode_count_per_variant": collection.episode_count_per_variant,
        "variant_order": list(VARIANT_IDS),
    }
    episode_level = {
        **shared,
        "artifact_type": "t4_ablation_episode_level",
        "records": list(collection.records),
        "paired_contributions": paired_contributions,
    }
    aggregate = {
        **shared,
        "artifact_type": "t4_ablation_aggregate",
        "metric_definitions": matrix["metrics"],
        "statistics": {
            **matrix["statistics"],
            "bootstrap_method": "paired_percentile",
        },
        "variants": variant_summaries,
        "comparisons": comparison_summaries,
        "control_drift": _control_drift(matrix, variant_summaries),
    }
    return episode_level, aggregate


def validate_analysis_outputs(
    matrix: dict[str, Any], episode_level: Any, aggregate: Any
) -> None:
    if not isinstance(episode_level, dict):
        raise ContractError("episode-level artifact must be an object")
    if set(episode_level) != {
        "schema_version",
        "matrix_id",
        "matrix_sha256",
        "analysis_status",
        "evidence_kind",
        "runtime_policy",
        "execution_target",
        "runner_commit_sha",
        "dataset_sha256",
        "episode_manifest_sha256",
        "episode_count_per_variant",
        "variant_order",
        "artifact_type",
        "records",
        "paired_contributions",
    }:
        raise ContractError("episode-level artifact keys are not exact")
    expected_episode, expected_aggregate = build_analysis_outputs(
        matrix, episode_level["records"]
    )
    if not json_values_equal(episode_level, expected_episode):
        raise ContractError("episode-level artifact is not a deterministic recomputation")
    if not json_values_equal(aggregate, expected_aggregate):
        raise ContractError("aggregate artifact is not a deterministic recomputation")


def analysis_manifest(
    episode_level: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "t4_ablation_analysis_manifest",
        "analysis_status": aggregate["analysis_status"],
        "episode_level_path": "episode_level.json",
        "episode_level_sha256": canonical_sha256(episode_level),
        "aggregate_path": "aggregate.json",
        "aggregate_sha256": canonical_sha256(aggregate),
        "resource_use": "none",
        "online_execution": False,
    }
