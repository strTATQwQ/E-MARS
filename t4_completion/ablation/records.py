"""Strict episode-record validation and cross-arm pairing for T4.6."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import (
    ContractError,
    ORACLE_UNSTABLE_REASONS,
    VARIANT_IDS,
    canonical_sha256,
    expect_exact_keys,
    expect_int,
    expect_number,
    expect_safe_path,
    expect_sha256,
    resolve_variant_configs,
    strict_json_loads,
    validate_matrix,
    variant_config_hashes,
)


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TERMINATION_REASONS = {
    "success",
    "failure",
    "timeout",
    "stuck",
    "collision",
    "fall",
    "stale",
    "reset_contamination",
    "error",
}
_EFFECTIVE_SOURCES = {
    "model_stop",
    "oracle_distance",
    "environment",
    "timeout",
    "stuck",
    "safety",
    "error",
}
_STOP_STABILITY = {"stable", "not_observed", "missing", "premature", "oscillatory"}
_ORACLE_REASON_TO_STABILITY = {
    "stop_unstable_missing": "missing",
    "stop_unstable_premature": "premature",
    "stop_unstable_oscillatory": "oscillatory",
}


@dataclass(frozen=True)
class ValidatedCollection:
    records: tuple[dict[str, Any], ...]
    by_variant: dict[str, tuple[dict[str, Any], ...]]
    dataset_sha256: str
    episode_manifest_sha256: str
    evidence_kind: str
    execution_target: str
    runner_commit_sha: str
    episode_count_per_variant: int


def _expect_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ContractError(f"{label} must be a nonempty, trimmed string")
    return value


def _expect_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{label} must be boolean")
    return value


def _expect_nullable_number(
    value: Any, label: str, *, minimum: float | None = None
) -> float | None:
    if value is None:
        return None
    return expect_number(value, label, minimum=minimum)


def _validate_latency_summary(value: Any, label: str) -> None:
    summary = expect_exact_keys(
        value, {"count", "mean", "p50", "p95", "max"}, label
    )
    count = expect_int(summary["count"], f"{label}.count", minimum=0)
    names = ("mean", "p50", "p95", "max")
    converted = {
        name: _expect_nullable_number(
            summary[name], f"{label}.{name}", minimum=0.0
        )
        for name in names
    }
    if count == 0:
        if any(converted[name] is not None for name in names):
            raise ContractError(f"{label} with count=0 requires all summaries null")
        return
    if any(converted[name] is None for name in names):
        raise ContractError(f"{label} with count>0 requires all summaries")
    assert all(converted[name] is not None for name in names)
    if converted["p50"] > converted["p95"] or converted["p95"] > converted["max"]:
        raise ContractError(f"{label} must satisfy p50 <= p95 <= max")
    if converted["mean"] > converted["max"]:
        raise ContractError(f"{label}.mean cannot exceed max")


def _variant_by_id(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["variant_id"]: item for item in matrix["variants"]}


def _validate_episode_record(
    record: Any,
    matrix: dict[str, Any],
    expected_config_hashes: dict[str, str],
) -> dict[str, Any]:
    top_keys = {
        "schema_version",
        "matrix_id",
        "matrix_sha256",
        "variant_id",
        "variant_config_sha256",
        "dataset_sha256",
        "episode_id",
        "episode_ordinal",
        "seed",
        "evidence_kind",
        "runtime_policy",
        "execution_target",
        "official_metrics",
        "diagnostics",
        "termination",
        "activation",
        "safety",
        "warnings",
        "provenance",
    }
    value = expect_exact_keys(record, top_keys, "episode_record")
    if expect_int(value["schema_version"], "episode_record.schema_version") != 1:
        raise ContractError("episode_record.schema_version must equal 1")
    if value["matrix_id"] != matrix["matrix_id"]:
        raise ContractError("episode_record.matrix_id does not match the matrix")
    expected_matrix_sha = canonical_sha256(matrix)
    if expect_sha256(value["matrix_sha256"], "episode_record.matrix_sha256") != expected_matrix_sha:
        raise ContractError("episode_record.matrix_sha256 does not match canonical matrix")

    variant_id = value["variant_id"]
    if variant_id not in VARIANT_IDS:
        raise ContractError(f"unknown episode_record.variant_id: {variant_id!r}")
    expected_config_sha = expected_config_hashes[variant_id]
    if (
        expect_sha256(
            value["variant_config_sha256"],
            "episode_record.variant_config_sha256",
        )
        != expected_config_sha
    ):
        raise ContractError("episode_record.variant_config_sha256 is not frozen")
    expect_sha256(value["dataset_sha256"], "episode_record.dataset_sha256")
    _expect_nonempty_string(value["episode_id"], "episode_record.episode_id")
    expect_int(value["episode_ordinal"], "episode_record.episode_ordinal", minimum=0)
    expect_int(value["seed"], "episode_record.seed", minimum=0)

    evidence_kind = value["evidence_kind"]
    if evidence_kind not in matrix["allowed_evidence_kinds"]:
        raise ContractError("episode_record.evidence_kind is not allowed")
    target_by_kind = {
        "synthetic": "offline_synthetic",
        "completion_sim": "isaac_simulation",
    }
    if value["execution_target"] != target_by_kind[evidence_kind]:
        raise ContractError(
            "episode_record.execution_target does not match evidence_kind"
        )
    if value["runtime_policy"] != "completion_sim":
        raise ContractError("episode_record.runtime_policy must be completion_sim")
    if value["execution_target"] in matrix["forbidden_targets"]:
        raise ContractError("episode_record targets forbidden hardware")

    official = expect_exact_keys(
        value["official_metrics"], {"sr", "os", "spl", "ne_m"}, "official_metrics"
    )
    for name in ("sr", "os"):
        if expect_int(official[name], f"official_metrics.{name}") not in {0, 1}:
            raise ContractError(f"official_metrics.{name} must be 0 or 1")
    expect_number(official["spl"], "official_metrics.spl", minimum=0.0, maximum=1.0)
    expect_number(official["ne_m"], "official_metrics.ne_m", minimum=0.0)
    if float(official["spl"]) > float(official["sr"]):
        raise ContractError("official_metrics.spl cannot exceed sr")
    if int(official["sr"]) > int(official["os"]):
        raise ContractError("official_metrics.sr cannot exceed os")

    diagnostics = expect_exact_keys(
        value["diagnostics"], {"stuck", "latency_ms"}, "diagnostics"
    )
    _expect_bool(diagnostics["stuck"], "diagnostics.stuck")
    latencies = expect_exact_keys(
        diagnostics["latency_ms"],
        {"system1", "system2", "end_to_end"},
        "diagnostics.latency_ms",
    )
    for name in ("system1", "system2", "end_to_end"):
        _validate_latency_summary(latencies[name], f"diagnostics.latency_ms.{name}")
    raw_warnings = value.get("warnings")
    recorder_partial_frame = isinstance(raw_warnings, list) and "recorder_partial_frame" in raw_warnings
    if latencies["end_to_end"]["count"] == 0 and not recorder_partial_frame:
        raise ContractError("end_to_end latency must contain observations")

    variants = _variant_by_id(matrix)
    factors = variants[variant_id]["factors"]
    termination = expect_exact_keys(
        value["termination"],
        {
            "configured_policy",
            "effective_source",
            "reason",
            "stop_stability",
            "model_stop_observed",
            "model_stop_suppressed_count",
            "oracle_mode_enabled",
            "terminated_by_oracle",
            "oracle_reason",
            "oracle_threshold_m",
            "oracle_threshold_source",
            "oracle_distance_source",
            "distance_at_termination_m",
        },
        "termination",
    )
    if termination["configured_policy"] != factors["termination_mode"]:
        raise ContractError("termination.configured_policy does not match variant")
    if termination["effective_source"] not in _EFFECTIVE_SOURCES:
        raise ContractError("termination.effective_source is invalid")
    if termination["reason"] not in _TERMINATION_REASONS:
        raise ContractError("termination.reason is invalid")
    if termination["reason"] in {"stale", "reset_contamination"}:
        raise ContractError("stale or reset-contaminated episodes are invalid evidence")
    if termination["stop_stability"] not in _STOP_STABILITY:
        raise ContractError("termination.stop_stability is invalid")
    _expect_bool(termination["model_stop_observed"], "termination.model_stop_observed")
    suppressed = expect_int(
        termination["model_stop_suppressed_count"],
        "termination.model_stop_suppressed_count",
        minimum=0,
    )
    oracle_enabled = _expect_bool(
        termination["oracle_mode_enabled"], "termination.oracle_mode_enabled"
    )
    terminated_by_oracle = _expect_bool(
        termination["terminated_by_oracle"], "termination.terminated_by_oracle"
    )
    observed_expected = termination["stop_stability"] in {
        "stable",
        "premature",
        "oscillatory",
    }
    if termination["model_stop_observed"] != observed_expected:
        raise ContractError("STOP stability and model_stop_observed disagree")
    oracle_fields = (
        "oracle_reason",
        "oracle_threshold_m",
        "oracle_threshold_source",
        "oracle_distance_source",
        "distance_at_termination_m",
    )
    expected_oracle = factors["termination_mode"] == "oracle_termination"
    if oracle_enabled != expected_oracle:
        raise ContractError("termination.oracle_mode_enabled does not match variant")
    if not expected_oracle:
        if terminated_by_oracle or any(termination[name] is not None for name in oracle_fields):
            raise ContractError("model_stop arm cannot claim oracle termination evidence")
        if termination["effective_source"] == "oracle_distance" or suppressed != 0:
            raise ContractError("model_stop arm cannot suppress STOP for an oracle")
    else:
        threshold = expect_number(
            termination["oracle_threshold_m"],
            "termination.oracle_threshold_m",
            minimum=0.000001,
        )
        if threshold != float(matrix["oracle_termination_contract"]["threshold_m"]):
            raise ContractError("oracle threshold value is not frozen")
        if (
            termination["oracle_threshold_source"]
            != matrix["oracle_termination_contract"]["threshold_source"]
        ):
            raise ContractError("oracle threshold source is not frozen")
        if (
            termination["oracle_distance_source"]
            != matrix["oracle_termination_contract"]["distance_source"]
        ):
            raise ContractError("oracle distance source is not evaluator ground truth")
        distance = _expect_nullable_number(
            termination["distance_at_termination_m"],
            "termination.distance_at_termination_m",
            minimum=0.0,
        )
        if terminated_by_oracle:
            if termination["effective_source"] != "oracle_distance":
                raise ContractError("oracle marker requires oracle_distance source")
            reason = termination["oracle_reason"]
            if reason not in ORACLE_UNSTABLE_REASONS:
                raise ContractError("oracle marker requires a frozen STOP instability reason")
            if termination["stop_stability"] != _ORACLE_REASON_TO_STABILITY[reason]:
                raise ContractError("oracle reason and STOP stability disagree")
            if reason == "stop_unstable_missing":
                if termination["model_stop_observed"] or suppressed != 0:
                    raise ContractError("missing STOP oracle marker has inconsistent STOP evidence")
            elif not termination["model_stop_observed"] or suppressed == 0:
                raise ContractError("suppressed unstable STOP requires observed STOP evidence")
            if distance is None or distance > threshold:
                raise ContractError("oracle termination distance exceeds frozen threshold")
            if termination["reason"] != "success":
                raise ContractError("oracle termination must be marked as success")
            if int(official["sr"]) != 1 or int(official["os"]) != 1:
                raise ContractError("oracle termination requires SR=OS=1")
            if not math.isclose(
                float(official["ne_m"]), distance, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ContractError(
                    "oracle evaluator distance must match official navigation error"
                )
        else:
            if termination["effective_source"] == "oracle_distance":
                raise ContractError("oracle_distance source requires terminated_by_oracle")
            if termination["oracle_reason"] is not None:
                raise ContractError("unused oracle termination cannot claim an oracle reason")
            if termination["stop_stability"] in {"missing", "premature", "oscillatory"}:
                raise ContractError("unstable STOP in oracle arm requires an oracle marker")
            if suppressed != 0:
                raise ContractError("unused oracle termination cannot suppress model STOP")
    if (
        termination["effective_source"] == "model_stop"
        and not termination["model_stop_observed"]
    ):
        raise ContractError("model_stop source requires observed model STOP")

    activation = expect_exact_keys(
        value["activation"],
        {
            "system1_count",
            "system2_count",
            "oracle_high_level_count",
            "oracle_local_path_count",
            "trajectory_transform_count",
            "observed_trajectory_mode",
            "history_frame_count",
            "history_reset_count",
            "recovery_trigger_count",
            "visual_frame_count",
            "observed_view_mode",
            "visual_input_only",
        },
        "activation",
    )
    count_names = (
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
    counts = {
        name: expect_int(activation[name], f"activation.{name}", minimum=0)
        for name in count_names
    }
    system_mode = factors["system_mode"]
    if system_mode == "full_system1_system2":
        if counts["oracle_high_level_count"] or counts["oracle_local_path_count"]:
            raise ContractError("full system arm cannot contain oracle system activation")
        if (counts["system1_count"] > 0) != (latencies["system1"]["count"] > 0):
            raise ContractError("full system arm has inconsistent System1 latency evidence")
        if (counts["system2_count"] > 0) != (latencies["system2"]["count"] > 0):
            raise ContractError("full system arm has inconsistent System2 latency evidence")
    elif system_mode == "oracle_high_level_system1":
        if counts["system2_count"] or counts["oracle_local_path_count"]:
            raise ContractError("oracle high-level arm activated the wrong system path")
        if (counts["system1_count"] > 0) != (latencies["system1"]["count"] > 0) or latencies["system2"]["count"] != 0:
            raise ContractError("oracle high-level arm has inconsistent model latency evidence")
    else:
        if counts["system1_count"] or counts["oracle_high_level_count"]:
            raise ContractError("oracle local-path arm activated the wrong system path")
        if (counts["system2_count"] > 0) != (latencies["system2"]["count"] > 0) or latencies["system1"]["count"] != 0:
            raise ContractError("oracle local-path arm has inconsistent model latency evidence")

    transformed = factors["trajectory_mode"] != "full_trajectory"
    if not transformed and counts["trajectory_transform_count"] > 0:
        raise ContractError("full trajectory arm cannot contain trajectory transforms")
    if activation["observed_trajectory_mode"] != factors["trajectory_mode"]:
        raise ContractError("observed trajectory mode does not match variant")
    if (factors["history_mode"] == "on") != (counts["history_frame_count"] > 0):
        raise ContractError("history activation does not match variant")
    if factors["recovery_mode"] == "off" and counts["recovery_trigger_count"] != 0:
        raise ContractError("recovery_off arm cannot contain recovery triggers")
    if counts["visual_frame_count"] == 0:
        raise ContractError("visual view arm requires observed visual frames")
    if activation["observed_view_mode"] != factors["view_mode"]:
        raise ContractError("observed view does not match variant")
    if _expect_bool(activation["visual_input_only"], "activation.visual_input_only") is not True:
        raise ContractError("view ablation must remain visual-input-only")

    safety = expect_exact_keys(
        value["safety"],
        {
            "collision_count",
            "fall_count",
            "stale_command_count",
            "reset_contamination_count",
            "bounded_velocity_violation_count",
            "simulation_estop_available",
        },
        "safety",
    )
    safety_counts = {
        name: expect_int(safety[name], f"safety.{name}", minimum=0)
        for name in (
            "collision_count",
            "fall_count",
            "stale_command_count",
            "reset_contamination_count",
            "bounded_velocity_violation_count",
        )
    }
    if safety_counts["stale_command_count"] != 0:
        raise ContractError("stale commands are forbidden")
    if safety_counts["reset_contamination_count"] != 0:
        raise ContractError("reset contamination is forbidden")
    if safety_counts["bounded_velocity_violation_count"] != 0:
        raise ContractError("bounded velocity violations are forbidden")
    if _expect_bool(safety["simulation_estop_available"], "safety.simulation_estop_available") is not True:
        raise ContractError("completion_sim requires a simulation estop")

    reason = termination["reason"]
    success = reason == "success"
    if int(official["sr"]) != int(success):
        raise ContractError("official SR must equal the success termination outcome")
    if success and float(official["spl"]) <= 0.0:
        raise ContractError("successful episode requires positive SPL")
    if bool(diagnostics["stuck"]) != (reason == "stuck"):
        raise ContractError("stuck diagnostic must match the termination reason")
    if reason == "collision" and safety_counts["collision_count"] == 0:
        raise ContractError("collision termination requires collision evidence")
    if reason == "fall" and safety_counts["fall_count"] == 0:
        raise ContractError("fall termination requires fall evidence")
    source_reason = {
        "timeout": "timeout",
        "stuck": "stuck",
        "error": "error",
    }
    if reason in source_reason and termination["effective_source"] != source_reason[reason]:
        raise ContractError("termination reason and effective source disagree")
    if reason in {"collision", "fall"} and termination["effective_source"] != "safety":
        raise ContractError("safety termination requires safety effective source")

    warnings = value["warnings"]
    if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise ContractError("warnings must be an array of strings")
    if len(warnings) != len(set(warnings)):
        raise ContractError("warnings must be unique")
    unknown_warnings = set(warnings) - set(
        matrix["completion_sim_invariants"]["warning_allowlist"]
    )
    if unknown_warnings:
        raise ContractError(f"warnings are not completion_sim allowlisted: {sorted(unknown_warnings)}")

    provenance = expect_exact_keys(
        value["provenance"],
        {
            "run_id",
            "runner_commit_sha",
            "episode_manifest_sha256",
            "source_result_path",
        },
        "provenance",
    )
    _expect_nonempty_string(provenance["run_id"], "provenance.run_id")
    if (
        not isinstance(provenance["runner_commit_sha"], str)
        or not _GIT_SHA_RE.fullmatch(provenance["runner_commit_sha"])
    ):
        raise ContractError("provenance.runner_commit_sha must be a 40-character Git SHA")
    expect_sha256(
        provenance["episode_manifest_sha256"],
        "provenance.episode_manifest_sha256",
    )
    expect_safe_path(provenance["source_result_path"], "provenance.source_result_path")
    return value


def validate_episode_record(record: Any, matrix: dict[str, Any]) -> dict[str, Any]:
    validate_matrix(matrix)
    return _validate_episode_record(record, matrix, variant_config_hashes(matrix))


def load_episode_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read {path.as_posix()}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        value = strict_json_loads(line, label=f"{path.as_posix()}:{line_number}")
        if not isinstance(value, dict):
            raise ContractError(f"{path.as_posix()}:{line_number} must be an object")
        records.append(value)
    if not records:
        raise ContractError("episode JSONL contains no records")
    return records


def pairing_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        record["dataset_sha256"],
        record["episode_id"],
        record["episode_ordinal"],
        record["seed"],
    )


def validate_episode_collection(
    records: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    matrix: dict[str, Any],
) -> ValidatedCollection:
    validate_matrix(matrix)
    if not isinstance(records, (list, tuple)) or not records:
        raise ContractError("episode collection must be nonempty")
    expected_config_hashes = variant_config_hashes(matrix)
    checked = [
        _validate_episode_record(record, matrix, expected_config_hashes)
        for record in records
    ]
    by_variant_lists: dict[str, list[dict[str, Any]]] = {
        variant_id: [] for variant_id in VARIANT_IDS
    }
    for record in checked:
        by_variant_lists[record["variant_id"]].append(record)
    missing = [variant_id for variant_id, rows in by_variant_lists.items() if not rows]
    if missing:
        raise ContractError(f"episode collection misses variants: {missing}")

    minimum = matrix["pairing"]["minimum_episode_count"]
    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    baseline_keys: tuple[tuple[str, str, int, int], ...] | None = None
    baseline_manifest: str | None = None
    run_ids: list[str] = []
    for variant_id in VARIANT_IDS:
        rows = by_variant_lists[variant_id]
        keys = [pairing_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise ContractError(f"duplicate paired episode in variant {variant_id}")
        episode_ids = [row["episode_id"] for row in rows]
        if len(episode_ids) != len(set(episode_ids)):
            raise ContractError(f"duplicate episode_id in variant {variant_id}")
        ordered = tuple(sorted(rows, key=lambda row: row["episode_ordinal"]))
        ordinals = [row["episode_ordinal"] for row in ordered]
        if ordinals != list(range(len(ordered))):
            raise ContractError(f"episode ordinals are not contiguous in {variant_id}")
        if len(ordered) < minimum:
            raise ContractError(
                f"variant {variant_id} has {len(ordered)} episodes; requires >= {minimum}"
            )
        ordered_keys = tuple(pairing_key(row) for row in ordered)
        manifests = {
            row["provenance"]["episode_manifest_sha256"] for row in ordered
        }
        if len(manifests) != 1:
            raise ContractError(f"variant {variant_id} mixes episode manifests")
        manifest = next(iter(manifests))
        variant_run_ids = {row["provenance"]["run_id"] for row in ordered}
        if len(variant_run_ids) != 1:
            raise ContractError(f"variant {variant_id} mixes run IDs")
        run_ids.append(next(iter(variant_run_ids)))
        if baseline_keys is None:
            baseline_keys = ordered_keys
            baseline_manifest = manifest
        else:
            if ordered_keys != baseline_keys:
                raise ContractError(
                    f"variant {variant_id} does not use the same paired episode order"
                )
            if manifest != baseline_manifest:
                raise ContractError(
                    f"variant {variant_id} uses a different episode manifest"
                )
        normalized[variant_id] = ordered

    variants = _variant_by_id(matrix)
    for variant_id, rows in normalized.items():
        factors = variants[variant_id]["factors"]
        totals = {
            name: sum(row["activation"][name] for row in rows)
            for name in (
                "system1_count",
                "system2_count",
                "oracle_high_level_count",
                "oracle_local_path_count",
                "trajectory_transform_count",
            )
        }
        if factors["system_mode"] == "full_system1_system2":
            if totals["system1_count"] == 0 and totals["system2_count"] == 0:
                raise ContractError(
                    f"variant {variant_id} lacks aggregate full-system activation"
                )
            # The dedicated system replacement block must exercise both model
            # paths because its two comparisons attribute their respective
            # gaps.  On other intervention axes, collapse to one observed path
            # can itself be the treatment effect (history_off did this online)
            # and must remain analyzable rather than be conditioned away.
            if variants[variant_id]["comparison_block"] == "system" and (
                totals["system1_count"] == 0 or totals["system2_count"] == 0
            ):
                raise ContractError(
                    f"variant {variant_id} lacks aggregate System1/System2 activation"
                )
        elif factors["system_mode"] == "oracle_high_level_system1":
            if totals["system1_count"] == 0 or totals["oracle_high_level_count"] == 0:
                raise ContractError(
                    f"variant {variant_id} lacks aggregate oracle high-level activation"
                )
        elif totals["system2_count"] == 0 or totals["oracle_local_path_count"] == 0:
            raise ContractError(
                f"variant {variant_id} lacks aggregate oracle local-path activation"
            )
        if (
            factors["trajectory_mode"] != "full_trajectory"
            and totals["trajectory_transform_count"] == 0
        ):
            raise ContractError(
                f"variant {variant_id} lacks aggregate trajectory transform activation"
            )

    dataset_hashes = {record["dataset_sha256"] for record in checked}
    evidence_kinds = {record["evidence_kind"] for record in checked}
    targets = {record["execution_target"] for record in checked}
    runner_commits = {record["provenance"]["runner_commit_sha"] for record in checked}
    source_paths = [record["provenance"]["source_result_path"] for record in checked]
    if len(dataset_hashes) != 1:
        raise ContractError("episode collection mixes dataset SHA-256 values")
    if len(evidence_kinds) != 1:
        raise ContractError("episode collection mixes evidence kinds")
    if len(targets) != 1:
        raise ContractError("episode collection mixes execution targets")
    if len(runner_commits) != 1:
        raise ContractError("episode collection mixes runner commits")
    if len(run_ids) != len(set(run_ids)):
        raise ContractError("each variant must use a distinct run ID")
    if len(source_paths) != len(set(source_paths)):
        raise ContractError("source_result_path must be unique per episode record")
    counts = {len(rows) for rows in normalized.values()}
    if len(counts) != 1:
        raise ContractError("variants do not have the same episode count")
    ordered_records = tuple(
        row for variant_id in VARIANT_IDS for row in normalized[variant_id]
    )
    assert baseline_manifest is not None
    return ValidatedCollection(
        records=ordered_records,
        by_variant=normalized,
        dataset_sha256=next(iter(dataset_hashes)),
        episode_manifest_sha256=baseline_manifest,
        evidence_kind=next(iter(evidence_kinds)),
        execution_target=next(iter(targets)),
        runner_commit_sha=next(iter(runner_commits)),
        episode_count_per_variant=next(iter(counts)),
    )


def _build_synthetic_record_template(
    matrix: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    variant_id: str,
    episode_ordinal: int,
) -> dict[str, Any]:
    if variant_id not in configs:
        raise ContractError(f"unknown variant_id: {variant_id}")
    factors = configs[variant_id]["factors"]
    system_mode = factors["system_mode"]
    activation = {
        "system1_count": 5 if system_mode != "system2_oracle_local_path" else 0,
        "system2_count": 2 if system_mode != "oracle_high_level_system1" else 0,
        "oracle_high_level_count": 2 if system_mode == "oracle_high_level_system1" else 0,
        "oracle_local_path_count": 5 if system_mode == "system2_oracle_local_path" else 0,
        "trajectory_transform_count": 0
        if factors["trajectory_mode"] == "full_trajectory"
        else 5,
        "observed_trajectory_mode": factors["trajectory_mode"],
        "history_frame_count": 8 if factors["history_mode"] == "on" else 0,
        "history_reset_count": 1,
        "recovery_trigger_count": 1 if factors["recovery_mode"] == "on" else 0,
        "visual_frame_count": 12,
        "observed_view_mode": factors["view_mode"],
        "visual_input_only": True,
    }
    oracle_enabled = factors["termination_mode"] == "oracle_termination"
    latency = lambda value: {  # noqa: E731 - compact immutable fixture builder
        "count": 5,
        "mean": value,
        "p50": value,
        "p95": value + 1.0,
        "max": value + 2.0,
    }
    no_latency = {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "schema_version": 1,
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": canonical_sha256(matrix),
        "variant_id": variant_id,
        "variant_config_sha256": canonical_sha256(configs[variant_id]),
        "dataset_sha256": "a" * 64,
        "episode_id": f"synthetic-{episode_ordinal:03d}",
        "episode_ordinal": episode_ordinal,
        "seed": 1000 + episode_ordinal,
        "evidence_kind": "synthetic",
        "runtime_policy": "completion_sim",
        "execution_target": "offline_synthetic",
        "official_metrics": {"sr": 1, "os": 1, "spl": 0.5, "ne_m": 1.0},
        "diagnostics": {
            "stuck": False,
            "latency_ms": {
                "system1": no_latency
                if system_mode == "system2_oracle_local_path"
                else latency(10.0),
                "system2": no_latency
                if system_mode == "oracle_high_level_system1"
                else latency(20.0),
                "end_to_end": latency(35.0),
            },
        },
        "termination": {
            "configured_policy": factors["termination_mode"],
            "effective_source": "oracle_distance" if oracle_enabled else "model_stop",
            "reason": "success",
            "stop_stability": "missing" if oracle_enabled else "stable",
            "model_stop_observed": not oracle_enabled,
            "model_stop_suppressed_count": 0,
            "oracle_mode_enabled": oracle_enabled,
            "terminated_by_oracle": oracle_enabled,
            "oracle_reason": "stop_unstable_missing" if oracle_enabled else None,
            "oracle_threshold_m": 2.5 if oracle_enabled else None,
            "oracle_threshold_source": "frozen_evaluator_success_radius"
            if oracle_enabled
            else None,
            "oracle_distance_source": "evaluator_ground_truth"
            if oracle_enabled
            else None,
            "distance_at_termination_m": 1.0 if oracle_enabled else None,
        },
        "activation": activation,
        "safety": {
            "collision_count": 0,
            "fall_count": 0,
            "stale_command_count": 0,
            "reset_contamination_count": 0,
            "bounded_velocity_violation_count": 0,
            "simulation_estop_available": True,
        },
        "warnings": [],
        "provenance": {
            "run_id": f"synthetic-{variant_id}",
            "runner_commit_sha": "b" * 40,
            "episode_manifest_sha256": "c" * 64,
            "source_result_path": f"synthetic/{variant_id}/{episode_ordinal:03d}.json",
        },
    }


def build_synthetic_record_template(
    matrix: dict[str, Any], variant_id: str, episode_ordinal: int
) -> dict[str, Any]:
    """Return one valid-by-construction, explicitly synthetic test record."""

    validate_matrix(matrix)
    configs = resolve_variant_configs(matrix)
    return _build_synthetic_record_template(
        matrix, configs, variant_id, episode_ordinal
    )


def build_synthetic_collection(
    matrix: dict[str, Any], *, episode_count: int = 20
) -> list[dict[str, Any]]:
    """Build all frozen arms efficiently for offline contract tests only."""

    validate_matrix(matrix)
    if isinstance(episode_count, bool) or not isinstance(episode_count, int):
        raise ContractError("synthetic episode_count must be an integer")
    if episode_count < matrix["pairing"]["minimum_episode_count"]:
        raise ContractError("synthetic episode_count is below the frozen minimum")
    configs = resolve_variant_configs(matrix)
    return [
        _build_synthetic_record_template(matrix, configs, variant_id, ordinal)
        for variant_id in VARIANT_IDS
        for ordinal in range(episode_count)
    ]
