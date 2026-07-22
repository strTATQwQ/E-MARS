#!/usr/bin/env python3
"""Validate and aggregate the disjoint T5 final-pilot A10+B10 results.

Integrity and promotion are intentionally separate.  A complete, clean 0/20
run is valid evidence (integrity PASS) but is not eligible for Golden Bundle
promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION_MANIFEST = (
    ROOT / "configs" / "internnav_t5" / "t5_24h_execution_manifest.json"
)
DEFAULT_PILOT_MANIFEST = (
    ROOT / "configs" / "internnav_t3" / "model_pilot_episode_manifest.json"
)
EXPECTED_SOURCE_GZIP_SHA256 = (
    "f09b11004d15d579620ba2b2769dcbf863efe4cf04d9e404f071f66244463f8b"
)
FROZEN_LANE_KEYS = {
    "a": [
        "6898_1741",
        "6292_1573",
        "5840_1474",
        "6842_1720",
        "1420_364",
        "4084_1003",
        "583_145",
        "1803_448",
        "6623_1657",
        "2613_625",
    ],
    "b": [
        "5627_1417",
        "4182_1027",
        "4009_976",
        "654_157",
        "4943_1255",
        "6982_1765",
        "3542_877",
        "6157_1561",
        "2564_610",
        "2853_703",
    ],
}
FROZEN_KEYS = FROZEN_LANE_KEYS["a"] + FROZEN_LANE_KEYS["b"]
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MINIMUM_PROMOTION_SUCCESS_COUNT = 1


def _file_sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if path.is_symlink():
        return None, "symlink is not accepted"
    if not path.is_file():
        return None, "regular file is missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def _checks_pass(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _binary(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    numeric = _finite(value, label)
    if numeric not in {0.0, 1.0}:
        raise ValueError(f"{label} must be boolean or 0/1")
    return bool(numeric)


def _metric_candidates(record: dict[str, Any], names: Iterable[str]) -> list[Any]:
    values = [record[name] for name in names if name in record]
    official = record.get("official_metrics")
    if isinstance(official, dict):
        values.extend(official[name] for name in names if name in official)
    return values


def _one_number(record: dict[str, Any], label: str, names: Iterable[str]) -> float:
    values = _metric_candidates(record, names)
    if not values:
        raise ValueError(f"episode record is missing explicit {label}")
    converted = [_finite(value, label) for value in values]
    if any(value != converted[0] for value in converted[1:]):
        raise ValueError(f"episode record has conflicting {label} values")
    return converted[0]


def _one_boolean(
    record: dict[str, Any], label: str, names: Iterable[str]
) -> bool:
    values = _metric_candidates(record, names)
    if not values:
        raise ValueError(f"episode record is missing explicit {label}")
    converted = [_binary(value, label) for value in values]
    if any(value != converted[0] for value in converted[1:]):
        raise ValueError(f"episode record has conflicting {label} values")
    return converted[0]


def _materialize_metric(
    record: dict[str, Any], expected_key: str, expected_episode_id: str
) -> dict[str, Any]:
    trajectory = record.get("trajectory_id")
    if not isinstance(trajectory, (str, int)) or not str(trajectory).strip():
        raise ValueError("per_episode record lacks trajectory_id")
    trajectory_text = str(trajectory).strip()
    trajectory_token = trajectory_text.rsplit("::", 1)[-1].rsplit("_", 1)[-1]
    if trajectory_text != expected_key and trajectory_token != expected_episode_id:
        raise ValueError(
            f"per_episode trajectory_id {trajectory_text!r} does not identify "
            f"ordered episode {expected_key!r}"
        )
    if "episode_key" in record and str(record["episode_key"]) != expected_key:
        raise ValueError("per_episode episode_key conflicts with ordered manifest")
    official = record.get("official_metrics")
    if not isinstance(official, dict):
        raise ValueError("per_episode record lacks official_metrics")
    success = _binary(official.get("sr"), "official SR")
    oracle = _binary(official.get("os"), "official OS")
    stuck_values = _metric_candidates(record, ("stuck",))
    reason_present = "termination_reason" in record
    reason = str(record.get("termination_reason", "")).strip().lower()
    if not stuck_values and not reason_present:
        raise ValueError("episode record is missing explicit stuck/termination_reason")
    converted_stuck = [_binary(value, "stuck") for value in stuck_values]
    if reason_present:
        converted_stuck.append(reason == "stuck")
    if any(value != converted_stuck[0] for value in converted_stuck[1:]):
        raise ValueError("episode record has conflicting stuck values")
    spl = _finite(official.get("spl"), "official SPL")
    ndtw = _finite(official.get("ndtw"), "official nDTW")
    ne = _finite(official.get("ne_m"), "official NE")
    if not 0.0 <= spl <= 1.0 or not 0.0 <= ndtw <= 1.0:
        raise ValueError("episode SPL and nDTW must be inside [0,1]")
    if ne < 0.0:
        raise ValueError("episode NE must be nonnegative")
    return {
        "episode_key": expected_key,
        "source_trajectory_id": trajectory_text,
        "success": success,
        "oracle_success": oracle,
        "stuck": converted_stuck[0],
        "SPL": spl,
        "nDTW": ndtw,
        "NE": ne,
        "source_record_sha256": hashlib.sha256(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _find_metrics(result_root: Path) -> Path:
    evaluator = result_root / "remote" / "x86" / "evaluator"
    candidates = sorted(
        path
        for path in evaluator.rglob("per_episode.json")
        if path.is_file() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise ValueError("exactly one evaluator per_episode.json is required")
    return candidates[0]


def _ordered_episode_contract(
    order: dict[str, Any], expected_keys: list[str], dataset_sha256: Any
) -> tuple[list[str], list[str]]:
    raw_keys = order.get("raw_episode_keys")
    pre_reverse = order.get("materialized_pre_reverse_episode_keys")
    ordered_keys = order.get("ordered_episode_keys")
    ordered_ids = order.get("ordered_episode_ids")
    expected_ids = [key.rsplit("_", 1)[-1] for key in expected_keys]
    if order.get("schema_version") != 1 or order.get("status") != "PASS":
        raise ValueError("ordered episode manifest did not PASS")
    if order.get("dataset_sha256") != dataset_sha256:
        raise ValueError("ordered episode manifest dataset SHA differs from binding")
    if order.get("dataset_episode_count") != len(expected_keys):
        raise ValueError("ordered episode manifest count differs from frozen lane")
    if raw_keys != expected_keys or pre_reverse != expected_keys:
        raise ValueError("ordered manifest raw/pre-reverse keys differ from frozen lane")
    if ordered_keys != list(reversed(expected_keys)):
        raise ValueError("ordered manifest does not seal the expected loader reversal")
    if ordered_ids != list(reversed(expected_ids)):
        raise ValueError("ordered manifest episode IDs differ from ordered keys")
    return list(ordered_keys), list(ordered_ids)


def _load_metrics(
    result_root: Path, ordered_keys: list[str], ordered_ids: list[str]
) -> tuple[
    list[dict[str, Any]], Path
]:
    path = _find_metrics(result_root)
    document, error = _load_object(path)
    if error is not None or document is None:
        raise ValueError(f"episode metrics are invalid: {error}")
    rows = document.get("episodes")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("episode metrics must contain an episodes object array")
    if len(rows) != len(ordered_keys):
        raise ValueError("episode metrics count differs from the frozen lane split")
    if document.get("completed_episode_count") != len(ordered_keys):
        raise ValueError("per_episode completed count differs from the lane split")
    if (
        "expected_episode_count" in document
        and document["expected_episode_count"] != len(ordered_keys)
    ):
        raise ValueError("per_episode expected count differs from the lane split")
    for name in ("expected_episode_count", "completed_episode_count"):
        if name in document and document[name] != len(ordered_keys):
            raise ValueError(f"episode metrics {name} differs from the lane split")
    materialized = [
        _materialize_metric(row, key, episode_id)
        for row, key, episode_id in zip(rows, ordered_keys, ordered_ids)
    ]
    return materialized, path


def _attach_runtime_command_age(
    runtime_metrics: dict[str, Any],
    metrics: list[dict[str, Any]],
    metrics_path: Path,
    result_root: Path,
) -> list[dict[str, Any]]:
    if runtime_metrics.get("status") != "PASS" or not _checks_pass(
        runtime_metrics.get("checks")
    ):
        raise ValueError("episode_runtime_metrics.json did not PASS")
    rows = runtime_metrics.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(metrics) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError("runtime metrics episode count differs from per_episode")
    relative_metrics = metrics_path.relative_to(result_root).as_posix()
    sources = runtime_metrics.get("sources")
    source_receipt = sources.get(relative_metrics) if isinstance(sources, dict) else None
    if (
        not isinstance(source_receipt, dict)
        or source_receipt.get("sha256") != _file_sha256(metrics_path)
    ):
        raise ValueError("runtime metrics do not seal the exact per_episode SHA")

    attached: list[dict[str, Any]] = []
    for index, (metric, runtime) in enumerate(zip(metrics, rows), start=1):
        if runtime.get("ordinal") != index:
            raise ValueError("runtime metrics ordinal/order differs from per_episode")
        runtime_trajectory = runtime.get("trajectory_id")
        if str(runtime_trajectory).strip() != metric["source_trajectory_id"]:
            raise ValueError("runtime trajectory_id/order differs from per_episode")
        controller = runtime.get("controller")
        all_updates = (
            controller.get("all_updates") if isinstance(controller, dict) else None
        )
        stats = (
            all_updates.get("command_age_sec")
            if isinstance(all_updates, dict)
            else None
        )
        if not isinstance(stats, dict):
            raise ValueError("runtime episode lacks all-updates command-age stats")
        count = stats.get("count")
        if type(count) is not int or count <= 0:
            raise ValueError("runtime episode has no command-age samples")
        minimum = _finite(stats.get("minimum"), "command-age minimum")
        mean = _finite(stats.get("mean"), "command-age mean")
        p95 = _finite(stats.get("p95"), "command-age p95")
        maximum = _finite(stats.get("maximum"), "command-age maximum")
        if (
            minimum < 0.0
            or not minimum <= mean <= maximum
            or not minimum <= p95 <= maximum
        ):
            raise ValueError("runtime command-age statistics are inconsistent")
        item = dict(metric)
        item.update(
            {
                "command_age_sec": mean,
                "command_age_sec_p95": p95,
                "command_age_sec_max": maximum,
                "command_age_sample_count": count,
                "command_age_source": (
                    "analysis/episode_runtime_metrics.json:"
                    "controller.all_updates.command_age_sec"
                ),
            }
        )
        attached.append(item)
    return attached


def _lane_evidence(
    result_root: Path,
    lane: str,
    expected_keys: list[str],
    expected_dataset_sha: Any,
) -> dict[str, Any]:
    result_root = result_root.resolve()
    required_paths = {
        "binding": result_root / "input_binding.json",
        "runtime": result_root / "fast_lane_summary.json",
        "final": result_root / "fast_lane_final_summary.json",
        "lease": result_root / "lease_release_summary.json",
        "dgx_status": result_root / "remote" / "dgx" / "lane_status.json",
        "x86_status": result_root / "remote" / "x86" / "isaac_status.json",
        "episode_order": result_root
        / "remote"
        / "x86"
        / "ordered_episode_manifest.json",
        "runtime_metrics": result_root
        / "analysis"
        / "episode_runtime_metrics.json",
    }
    documents: dict[str, dict[str, Any]] = {}
    source_errors: dict[str, str] = {}
    evidence_sha256: dict[str, str | None] = {}
    for name, path in required_paths.items():
        document, error = _load_object(path)
        documents[name] = document or {}
        evidence_sha256[name] = _file_sha256(path)
        if error is not None:
            source_errors[name] = error
    binding = documents["binding"]
    runtime = documents["runtime"]
    final = documents["final"]
    lease = documents["lease"]
    dgx_status = documents["dgx_status"]
    x86_status = documents["x86_status"]
    episode_order = documents["episode_order"]
    runtime_metrics = documents["runtime_metrics"]

    metrics: list[dict[str, Any]] = []
    metrics_path: Path | None = None
    ordered_keys: list[str] = []
    ordered_ids: list[str] = []
    try:
        ordered_keys, ordered_ids = _ordered_episode_contract(
            episode_order, expected_keys, binding.get("dataset_sha256")
        )
        metrics, metrics_path = _load_metrics(result_root, ordered_keys, ordered_ids)
        metrics = _attach_runtime_command_age(
            runtime_metrics, metrics, metrics_path, result_root
        )
        evidence_sha256["episode_metrics"] = _file_sha256(metrics_path)
    except (OSError, ValueError) as error:
        source_errors["episode_metrics"] = str(error)
        evidence_sha256["episode_metrics"] = None

    source_receipts = binding.get("source_receipts")
    source_receipts_valid = (
        isinstance(source_receipts, dict)
        and bool(source_receipts)
        and all(SHA256.fullmatch(str(value)) for value in source_receipts.values())
    )
    code_sha = runtime.get("code_ref_sha")
    resolution_sha = binding.get("candidate_resolution_sha256")
    map_sha = binding.get("static_map_manifest_sha256")
    zero_residual = (
        type(dgx_status.get("residual_count")) is int
        and dgx_status.get("residual_count") == 0
        and type(x86_status.get("residual_count")) is int
        and x86_status.get("residual_count") == 0
        and type(x86_status.get("socket_residual_count")) is int
        and x86_status.get("socket_residual_count") == 0
        and type(x86_status.get("clock_publishers_after_stop")) is int
        and x86_status.get("clock_publishers_after_stop") == 0
        and x86_status.get("lane_lock_released") is True
        and x86_status.get("shared_asset_lock_fd_released") is True
    )
    checks = {
        "result_root_regular": result_root.is_dir() and not result_root.is_symlink(),
        "all_required_sources_regular": not source_errors
        and all(value is not None for value in evidence_sha256.values()),
        "input_binding_pass": binding.get("status") == "PASS"
        and _checks_pass(binding.get("checks")),
        "runtime_summary_pass": runtime.get("status") == "PASS"
        and _checks_pass(runtime.get("checks")),
        "final_summary_pass": final.get("status") == "PASS"
        and _checks_pass(final.get("checks")),
        "lease_release_pass": lease.get("status") == "PASS"
        and _checks_pass(lease.get("checks")),
        "runtime_embeds_exact_binding": runtime.get("input_binding") == binding,
        "final_embeds_exact_runtime": final.get("runtime_summary") == runtime,
        "final_embeds_exact_lease_release": final.get("lease_release") == lease,
        "lane_identity_exact": binding.get("lane") == lane
        and runtime.get("lane") == lane,
        "execution_count_exact": binding.get("execution_episode_count") == 10,
        "execution_keys_exact": binding.get("execution_episode_keys")
        == expected_keys,
        "ordered_episode_contract_exact": ordered_keys
        == list(reversed(expected_keys))
        and ordered_ids == [key.rsplit("_", 1)[-1] for key in ordered_keys],
        "dataset_sha_valid": SHA256.fullmatch(str(binding.get("dataset_sha256")))
        is not None,
        "dataset_sha_matches_split_audit": binding.get("dataset_sha256")
        == expected_dataset_sha,
        "code_sha_valid_and_bound": SHA40.fullmatch(str(code_sha)) is not None
        and binding.get("code_ref_sha") == code_sha,
        "candidate_resolution_sha_valid_and_bound": SHA256.fullmatch(
            str(resolution_sha)
        )
        is not None
        and runtime.get("candidate_resolution_sha256") == resolution_sha,
        "static_map_sha_valid": SHA256.fullmatch(str(map_sha)) is not None,
        "source_receipts_valid": source_receipts_valid,
        "remote_statuses_exactly_embedded": runtime.get("dgx_status") == dgx_status
        and runtime.get("x86_status") == x86_status,
        "dgx_status_pass": dgx_status.get("status") == "PASS",
        "x86_status_pass": x86_status.get("status") == "PASS",
        "zero_resource_residual": zero_residual,
        "metrics_complete_and_ordered": len(metrics) == 10
        and [row["episode_key"] for row in metrics] == ordered_keys,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "lane": lane,
        "result_root": str(result_root),
        "execution_episode_keys": [row["episode_key"] for row in metrics],
        "authored_episode_keys": expected_keys,
        "ordered_episode_keys": ordered_keys,
        "dataset_sha256": binding.get("dataset_sha256"),
        "code_ref_sha": code_sha,
        "candidate_resolution_sha256": resolution_sha,
        "candidate_binding": binding.get("candidate_binding"),
        "static_map_manifest_sha256": map_sha,
        "metrics": metrics,
        "zero_residual": zero_residual,
        "checks": checks,
        "source_errors": source_errors,
        "evidence_sha256": evidence_sha256,
        "episode_metrics_relative_path": (
            metrics_path.relative_to(result_root).as_posix()
            if metrics_path is not None
            else None
        ),
    }


def _manifest_evidence(
    execution_manifest_path: Path,
    pilot_manifest_path: Path,
    expected_source_sha256: str,
) -> dict[str, Any]:
    execution, execution_error = _load_object(execution_manifest_path)
    pilot, pilot_error = _load_object(pilot_manifest_path)
    execution = execution or {}
    pilot = pilot or {}
    final = execution.get("final_evaluation")
    authored = final.get("pilot") if isinstance(final, dict) else None
    authored = authored if isinstance(authored, dict) else {}
    checks = {
        "manifest_sources_regular": execution_error is None
        and pilot_error is None
        and _file_sha256(execution_manifest_path) is not None
        and _file_sha256(pilot_manifest_path) is not None,
        "expected_source_sha_valid": SHA256.fullmatch(expected_source_sha256)
        is not None,
        "execution_lane_a_keys_exact": authored.get("lane_a_episode_keys")
        == FROZEN_LANE_KEYS["a"],
        "execution_lane_b_keys_exact": authored.get("lane_b_episode_keys")
        == FROZEN_LANE_KEYS["b"],
        "execution_aggregate_count_exact": authored.get("aggregate_episode_count")
        == 20,
        "held_out_tuning_forbidden": authored.get("held_out_tuning_forbidden")
        is True,
        "pilot_episode_count_exact": pilot.get("episode_count") == 20,
        "pilot_episode_keys_exact": pilot.get("episode_keys") == FROZEN_KEYS,
        "pilot_overlay_sha_exact": pilot.get("overlay_sha256")
        == expected_source_sha256,
        "frozen_lanes_disjoint": set(FROZEN_LANE_KEYS["a"]).isdisjoint(
            FROZEN_LANE_KEYS["b"]
        ),
        "frozen_union_exact": set(FROZEN_LANE_KEYS["a"])
        | set(FROZEN_LANE_KEYS["b"])
        == set(FROZEN_KEYS),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": {
            name: error
            for name, error in (
                ("execution_manifest", execution_error),
                ("pilot_manifest", pilot_error),
            )
            if error is not None
        },
        "sha256": {
            "execution_manifest": _file_sha256(execution_manifest_path),
            "pilot_manifest": _file_sha256(pilot_manifest_path),
        },
    }


def _split_evidence(
    split_audit_path: Path, expected_source_sha256: str
) -> dict[str, Any]:
    audit, error = _load_object(split_audit_path)
    audit = audit or {}
    source = audit.get("source")
    source = source if isinstance(source, dict) else {}
    manifests = audit.get("manifests")
    manifests = manifests if isinstance(manifests, dict) else {}
    execution_receipt = (
        manifests.get("execution")
        if isinstance(manifests.get("execution"), dict)
        else {}
    )
    pilot_receipt = (
        manifests.get("pilot") if isinstance(manifests.get("pilot"), dict) else {}
    )
    lanes = audit.get("lanes")
    lanes = lanes if isinstance(lanes, dict) else {}
    observed_a = lanes.get("a") if isinstance(lanes.get("a"), dict) else {}
    observed_b = lanes.get("b") if isinstance(lanes.get("b"), dict) else {}
    source_path = Path(str(source.get("dataset", "")))
    lane_paths = {
        "a": Path(str(observed_a.get("output_dataset", ""))),
        "b": Path(str(observed_b.get("output_dataset", ""))),
    }
    actual_lane_shas = {
        lane: _file_sha256(path) for lane, path in lane_paths.items()
    }
    checks = {
        "audit_source_regular": error is None
        and _file_sha256(split_audit_path) is not None,
        "audit_pass": audit.get("status") == "PASS"
        and _checks_pass(audit.get("checks")),
        "source_sha_exact": source.get("sha256") == expected_source_sha256,
        "source_file_sha_exact": _file_sha256(source_path)
        == expected_source_sha256,
        "source_count_exact": source.get("episode_count") == 20,
        "source_keys_exact": source.get("episode_keys") == FROZEN_KEYS,
        "lane_a_keys_exact": observed_a.get("episode_keys")
        == FROZEN_LANE_KEYS["a"],
        "lane_b_keys_exact": observed_b.get("episode_keys")
        == FROZEN_LANE_KEYS["b"],
        "lane_counts_exact": observed_a.get("episode_count") == 10
        and observed_b.get("episode_count") == 10,
        "lane_dataset_shas_valid": SHA256.fullmatch(
            str(observed_a.get("output_sha256"))
        )
        is not None
        and SHA256.fullmatch(str(observed_b.get("output_sha256"))) is not None,
        "lane_dataset_files_match_audit": actual_lane_shas["a"]
        == observed_a.get("output_sha256")
        and actual_lane_shas["b"] == observed_b.get("output_sha256"),
        "lane_keys_disjoint": set(observed_a.get("episode_keys", [])).isdisjoint(
            observed_b.get("episode_keys", [])
        ),
        "lane_keys_union_exact": set(observed_a.get("episode_keys", []))
        | set(observed_b.get("episode_keys", []))
        == set(FROZEN_KEYS),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "error": error,
        "audit_sha256": _file_sha256(split_audit_path),
        "source_gzip_sha256": source.get("sha256"),
        "lane_dataset_sha256": {
            "a": observed_a.get("output_sha256"),
            "b": observed_b.get("output_sha256"),
        },
        "actual_lane_dataset_sha256": actual_lane_shas,
        "manifest_sha256": {
            "execution": execution_receipt.get("sha256"),
            "pilot": pilot_receipt.get("sha256"),
        },
    }


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(metrics) != 20:
        return None
    success_count = sum(row["success"] for row in metrics)
    stuck_count = sum(row["stuck"] for row in metrics)
    oracle_values = [
        row["oracle_success"]
        for row in metrics
        if row.get("oracle_success") is not None
    ]
    command_ages = [row["command_age_sec"] for row in metrics]
    command_age_p95 = [row["command_age_sec_p95"] for row in metrics]
    command_age_maxima = [row["command_age_sec_max"] for row in metrics]
    return {
        "episode_count": len(metrics),
        "success_count": success_count,
        "SR": success_count / len(metrics),
        "oracle_success_present_count": len(oracle_values),
        "oracle_success_count": sum(oracle_values)
        if len(oracle_values) == len(metrics)
        else None,
        "OS": (sum(oracle_values) / len(metrics))
        if len(oracle_values) == len(metrics)
        else None,
        "SPL": _mean([row["SPL"] for row in metrics]),
        "nDTW": _mean([row["nDTW"] for row in metrics]),
        "NE": _mean([row["NE"] for row in metrics]),
        "stuck_count": stuck_count,
        "stuck_rate": stuck_count / len(metrics),
        "command_age_sec_mean": _mean(command_ages),
        "command_age_sec_episode_p95_mean": _mean(command_age_p95),
        "command_age_sec_episode_p95_max": max(command_age_p95),
        "command_age_sec_max": max(command_age_maxima),
        "command_age_aggregation": {
            "mean": "unweighted_mean_of_per_episode_all_updates_means",
            "p95": "mean_and_max_of_per_episode_all_updates_p95",
            "maximum": "maximum_of_per_episode_all_updates_maxima",
        },
    }


def finalize(
    lane_a_root: Path,
    lane_b_root: Path,
    split_audit_path: Path,
    *,
    execution_manifest_path: Path = DEFAULT_EXECUTION_MANIFEST,
    pilot_manifest_path: Path = DEFAULT_PILOT_MANIFEST,
    expected_source_sha256: str = EXPECTED_SOURCE_GZIP_SHA256,
) -> dict[str, Any]:
    lane_a_root = lane_a_root.resolve()
    lane_b_root = lane_b_root.resolve()
    split_audit_path = split_audit_path.resolve()
    execution_manifest_path = execution_manifest_path.resolve()
    pilot_manifest_path = pilot_manifest_path.resolve()
    manifests = _manifest_evidence(
        execution_manifest_path, pilot_manifest_path, expected_source_sha256
    )
    split = _split_evidence(split_audit_path, expected_source_sha256)
    lanes = {
        "a": _lane_evidence(
            lane_a_root,
            "a",
            FROZEN_LANE_KEYS["a"],
            split["lane_dataset_sha256"]["a"],
        ),
        "b": _lane_evidence(
            lane_b_root,
            "b",
            FROZEN_LANE_KEYS["b"],
            split["lane_dataset_sha256"]["b"],
        ),
    }
    observed_keys = {
        lane: evidence["execution_episode_keys"]
        for lane, evidence in lanes.items()
    }
    metrics_by_key = {
        row["episode_key"]: row
        for lane in ("a", "b")
        for row in lanes[lane]["metrics"]
    }
    both_metrics = (
        [metrics_by_key[key] for key in FROZEN_KEYS]
        if set(metrics_by_key) == set(FROZEN_KEYS)
        and sum(len(lanes[lane]["metrics"]) for lane in lanes) == 20
        else []
    )
    shared_checks = {
        "result_roots_distinct": lane_a_root != lane_b_root,
        "split_manifest_shas_match_current": split["manifest_sha256"]["execution"]
        == manifests["sha256"]["execution_manifest"]
        and split["manifest_sha256"]["pilot"]
        == manifests["sha256"]["pilot_manifest"],
        "lane_results_pass": all(lanes[lane]["status"] == "PASS" for lane in lanes),
        "observed_episode_keys_disjoint": set(observed_keys["a"]).isdisjoint(
            observed_keys["b"]
        ),
        "observed_episode_union_exact": set(observed_keys["a"])
        | set(observed_keys["b"])
        == set(FROZEN_KEYS),
        "observed_lane_orders_match_loader_contract": observed_keys["a"]
        == list(reversed(FROZEN_LANE_KEYS["a"]))
        and observed_keys["b"] == list(reversed(FROZEN_LANE_KEYS["b"])),
        "same_code_sha": lanes["a"]["code_ref_sha"] is not None
        and lanes["a"]["code_ref_sha"] == lanes["b"]["code_ref_sha"],
        "same_candidate_resolution_sha": lanes["a"][
            "candidate_resolution_sha256"
        ]
        is not None
        and lanes["a"]["candidate_resolution_sha256"]
        == lanes["b"]["candidate_resolution_sha256"],
        "same_candidate_binding": lanes["a"]["candidate_binding"] is not None
        and lanes["a"]["candidate_binding"] == lanes["b"]["candidate_binding"],
        "same_static_map_sha": lanes["a"]["static_map_manifest_sha256"]
        is not None
        and lanes["a"]["static_map_manifest_sha256"]
        == lanes["b"]["static_map_manifest_sha256"],
        "both_lanes_zero_residual": lanes["a"]["zero_residual"] is True
        and lanes["b"]["zero_residual"] is True,
        "twenty_metrics_available": len(both_metrics) == 20,
    }
    integrity_checks = {
        "frozen_manifests_pass": manifests["status"] == "PASS",
        "split_audit_pass": split["status"] == "PASS",
        **shared_checks,
    }
    integrity_pass = all(integrity_checks.values())
    aggregate = _aggregate(both_metrics) if integrity_pass else None
    promotion_eligible = (
        aggregate is not None
        and aggregate["success_count"] >= MINIMUM_PROMOTION_SUCCESS_COUNT
    )
    promotion_status = (
        "ELIGIBLE"
        if promotion_eligible
        else "NOT_ELIGIBLE"
        if integrity_pass
        else "NOT_EVALUABLE"
    )
    return {
        "schema_version": 1,
        "status": "PASS" if integrity_pass else "FAIL",
        "stage": "t5_final_pilot_disjoint_a10_b10",
        "integrity": {
            "status": "PASS" if integrity_pass else "FAIL",
            "checks": integrity_checks,
            "frozen_manifests": manifests,
            "split": split,
        },
        "promotion": {
            "status": promotion_status,
            "minimum_success_count": MINIMUM_PROMOTION_SUCCESS_COUNT,
            "observed_success_count": (
                aggregate["success_count"] if aggregate is not None else None
            ),
            "checks": {
                "integrity_pass": integrity_pass,
                "aggregate_success_at_least_one": promotion_eligible
                if integrity_pass
                else None,
            },
        },
        "aggregate": aggregate,
        "ordered_episode_keys": FROZEN_KEYS if integrity_pass else None,
        "lanes": lanes,
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-a-root", required=True, type=Path)
    parser.add_argument("--lane-b-root", required=True, type=Path)
    parser.add_argument("--split-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--execution-manifest", type=Path, default=DEFAULT_EXECUTION_MANIFEST
    )
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    arguments = parser.parse_args()
    try:
        report = finalize(
            arguments.lane_a_root,
            arguments.lane_b_root,
            arguments.split_audit,
            execution_manifest_path=arguments.execution_manifest,
            pilot_manifest_path=arguments.pilot_manifest,
        )
        _write_atomic(arguments.output.resolve(strict=False), report)
    except (OSError, ValueError) as error:
        print(f"final-pilot finalization failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["integrity"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
