#!/usr/bin/env python3
"""Validate D0.3 ancestry and finalize symmetric dual-Lane evidence.

This helper is deliberately offline: the online coordinator owns SSH, leases,
processes and cleanup.  It only consumes immutable JSON evidence and writes one
fail-closed receipt.  Keeping the statistical/sample and slowdown rules here
makes them directly fixture-testable without touching either simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when immutable D0 evidence does not satisfy the contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON evidence must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ContractError(f"cannot hash evidence {path}: {exc}") from exc


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def receipt_passes(value: dict[str, Any]) -> bool:
    checks = value.get("checks")
    return (
        value.get("status") == "PASS"
        and isinstance(checks, dict)
        and bool(checks)
        and all(item is True for item in checks.values())
    )


def _nested_runtime(final: dict[str, Any], lane: str) -> dict[str, Any]:
    runtime = final.get("runtime_summary")
    if not isinstance(runtime, dict):
        raise ContractError(f"D0.{1 if lane == 'a' else 2} has no runtime_summary")
    return runtime


def _duration_from_tree(x86_root: Path) -> float:
    contract = load_json(x86_root / "isaac_contract.json")
    status = load_json(x86_root / "isaac_status.json")
    try:
        started = float(contract["started_unix"])
        finished = float(status["finished_unix"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"missing finite x86 runtime timestamps under {x86_root}") from exc
    duration = finished - started
    if not math.isfinite(duration) or duration <= 0:
        raise ContractError(f"invalid x86 runtime duration under {x86_root}: {duration!r}")
    return duration


_LOWER_IS_BETTER = (
    "mean_inference_latency_sec",
    "mean_action_round_trip_latency_sec",
    "mean_nav2_resolution_latency_sec",
)


def _positive_finite(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"missing numeric performance signal {name}") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ContractError(f"performance signal {name} must be finite and positive")
    return number


def _performance_signals(dgx_root: Path, x86_root: Path) -> dict[str, Any]:
    client_path = dgx_root / "client" / "client_summary.json"
    client = load_json(client_path)
    if client.get("status") != "FINISHED" or int(client.get("step_count", 0)) <= 0:
        raise ContractError(f"client performance summary is incomplete: {client_path}")
    latency = {
        name: _positive_finite(client.get(name), name) for name in _LOWER_IS_BETTER
    }

    # A future x86 evaluator may archive controller_summary.json alongside its
    # attempt.  In the current split, the end-to-end controller lives on DGX
    # and measures the update stream originating from x86.  Both are accepted
    # as the same higher-is-better sensor/control throughput contract.
    x86_candidates = sorted(x86_root.rglob("controller_summary.json"))
    dgx_candidate = dgx_root / "onboard" / "controller_summary.json"
    if len(x86_candidates) > 1:
        raise ContractError("ambiguous x86 controller throughput summaries")
    if x86_candidates:
        controller_path = x86_candidates[0]
        source = "x86_controller_summary"
    elif dgx_candidate.is_file():
        controller_path = dgx_candidate
        source = "dgx_end_to_end_controller_summary_for_x86_sensor_stream"
    else:
        raise ContractError("missing controller/sensor throughput summary")
    controller = load_json(controller_path)
    if controller.get("status") != "FINISHED":
        raise ContractError(f"controller throughput summary is incomplete: {controller_path}")
    control_hz = _positive_finite(
        controller.get("measured_control_hz"), "measured_control_hz"
    )
    return {
        "lower_is_better": latency,
        "higher_is_better": {"measured_control_hz": control_hz},
        "sources": {
            "client_summary": str(client_path).replace("\\", "/"),
            "client_summary_sha256": sha256_file(client_path),
            "control_throughput": str(controller_path).replace("\\", "/"),
            "control_throughput_sha256": sha256_file(controller_path),
            "control_throughput_kind": source,
        },
    }


def _structured_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        )
    return sorted(set(files))


def _structured_prefix_count(value: Any, prefix: str) -> int:
    if isinstance(value, dict):
        return sum(_structured_prefix_count(item, prefix) for item in value.values())
    if isinstance(value, list):
        return sum(_structured_prefix_count(item, prefix) for item in value)
    return int(isinstance(value, str) and value.startswith(prefix))


def _scan_lane_identity(dgx_root: Path, x86_root: Path, lane: str) -> dict[str, Any]:
    own = f"{lane}::".encode("utf-8")
    opposite = (b"b::" if lane == "a" else b"a::")
    opposite_paths: list[str] = []
    opposite_byte_count = 0
    opposite_structured_count = 0
    own_count = 0
    for path in _structured_files(dgx_root, x86_root):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ContractError(f"cannot scan Lane identity evidence {path}: {exc}") from exc
        hits = raw.count(opposite)
        if hits:
            opposite_byte_count += hits
            opposite_paths.append(str(path).replace("\\", "/"))
        own_count += raw.count(own)
        # Parse every JSON/JSONL object too; malformed structured evidence is
        # fail-closed rather than silently skipped by a byte-only scan.
        try:
            if path.suffix.lower() == ".jsonl":
                values = []
                for line in raw.decode("utf-8").splitlines():
                    if line.strip():
                        values.append(json.loads(line))
            else:
                values = [json.loads(raw.decode("utf-8"))]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"malformed structured Lane evidence {path}: {exc}") from exc
        structured_hits = sum(
            _structured_prefix_count(value, opposite.decode("ascii")) for value in values
        )
        if structured_hits:
            opposite_structured_count += structured_hits
            normalized = str(path).replace("\\", "/")
            if normalized not in opposite_paths:
                opposite_paths.append(normalized)

    client_records = dgx_root / "client" / "client_records.jsonl"
    sensor_frames = x86_root / "health" / "sensor_frames.jsonl"
    try:
        client_raw = client_records.read_bytes()
        sensor_raw = sensor_frames.read_bytes()
    except OSError as exc:
        raise ContractError("missing real client/sensor identity records") from exc
    return {
        "opposite_prefix_match_count": opposite_byte_count
        + opposite_structured_count,
        "opposite_prefix_byte_match_count": opposite_byte_count,
        "opposite_prefix_structured_match_count": opposite_structured_count,
        "opposite_prefix_paths": opposite_paths,
        "own_prefix_match_count": own_count,
        "client_record_own_prefix_match_count": client_raw.count(own),
        "sensor_record_own_prefix_match_count": sensor_raw.count(own),
        "scanned_structured_file_count": len(_structured_files(dgx_root, x86_root)),
    }


def _episode_order(
    path: Path, prefix: str, *, allow_prebind_state_only: bool = False
) -> dict[str, Any]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read episode identity records {path}: {exc}") from exc
    observed: list[str] = []
    row_counts: dict[str, int] = {}
    malformed = 0
    ignored_prebind_state_only = 0
    bound = False
    for row in rows:
        if not isinstance(row, dict):
            malformed += 1
            continue
        episode = row.get("episode_id")
        bootstrap_sentinel = (
            isinstance(episode, str)
            and episode.startswith("bootstrap-episode-")
            and episode.removeprefix("bootstrap-episode-").isdigit()
        )
        if (
            allow_prebind_state_only
            and row.get("state_only") is True
            and (
                bootstrap_sentinel
                or (not bound and episode in (None, "", "uninitialized"))
            )
        ):
            # The continuous controller emits safe, motion-disabled warm-up
            # samples on initial start and each simulator reset.  Only the
            # strict bootstrap sentinel (plus a pre-bind empty/uninitialized
            # state) is outside fixed-five coverage.
            ignored_prebind_state_only += 1
            continue
        if not isinstance(episode, str) or not episode:
            malformed += 1
            continue
        if not episode.startswith(prefix):
            malformed += 1
        else:
            bound = True
        row_counts[episode] = row_counts.get(episode, 0) + 1
        if not observed or observed[-1] != episode:
            observed.append(episode)
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": sha256_file(path),
        "record_count": len(rows),
        "episode_row_counts": row_counts,
        "episode_transition_order": observed,
        "malformed_or_unprefixed_episode_count": malformed,
        "ignored_prebind_state_only_count": ignored_prebind_state_only,
    }


def _ordered_episode_contract(
    order_manifest_path: Path,
    readiness_evidence_path: Path,
    expected_episode_keys: list[str],
    expected_dataset_sha256: str,
) -> dict[str, Any]:
    for path, label in (
        (order_manifest_path, "episode-order manifest"),
        (readiness_evidence_path, "runtime-readiness evidence"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"{label} must be a regular non-symlink file")
    order = load_json(order_manifest_path)
    readiness = load_json(readiness_evidence_path)
    ordered_keys = order.get("ordered_episode_keys")
    ordered_ids = order.get("ordered_episode_ids")
    raw_keys = order.get("raw_episode_keys")
    pre_reverse_keys = order.get("materialized_pre_reverse_episode_keys")
    identity_by_key = {
        key: key.rsplit("_", 1)[-1] if "_" in key else ""
        for key in expected_episode_keys
    }
    readiness_order = readiness.get("model_episode_order_manifest")
    if not isinstance(readiness_order, dict):
        readiness_order = {}
    loader = order.get("loader_contract")
    if not isinstance(loader, dict):
        loader = {}
    checks = {
        "schema_and_status": order.get("schema_version") == 1
        and order.get("status") == "PASS",
        "dataset_sha256": isinstance(expected_dataset_sha256, str)
        and len(expected_dataset_sha256) == 64
        and order.get("dataset_sha256") == expected_dataset_sha256,
        "dataset_episode_count": order.get("dataset_episode_count")
        == len(expected_episode_keys)
        == 5,
        "raw_episode_keys_exact": raw_keys == expected_episode_keys,
        "ordered_episode_keys_exact_set": isinstance(ordered_keys, list)
        and len(ordered_keys) == len(expected_episode_keys)
        and len(set(ordered_keys)) == len(ordered_keys)
        and set(ordered_keys) == set(expected_episode_keys),
        "materialized_reverse_contract": isinstance(pre_reverse_keys, list)
        and isinstance(ordered_keys, list)
        and ordered_keys == list(reversed(pre_reverse_keys)),
        "ordered_episode_ids_match_keys": isinstance(ordered_ids, list)
        and isinstance(ordered_keys, list)
        and ordered_ids == [identity_by_key.get(key, "") for key in ordered_keys]
        and len(set(ordered_ids)) == len(ordered_ids)
        and all(ordered_ids),
        "fresh_loader_contract": loader.get("materializer")
        == "BasePathKeyEpisodeloader.path_key_data"
        and loader.get("fresh_resumable_order")
        == "reverse_materialized_path_keys"
        and loader.get("rank") == 0
        and loader.get("world_size") == 1,
        "runtime_readiness_pass": readiness.get("schema_version") == 1
        and readiness.get("status") == "PASS"
        and readiness.get("mode") == "model",
        "runtime_readiness_binds_manifest_sha256": readiness_order.get("sha256")
        == sha256_file(order_manifest_path),
    }
    if not all(checks.values()):
        raise ContractError(f"episode-order runtime contract failed: {checks}")
    return {
        "path": str(order_manifest_path).replace("\\", "/"),
        "sha256": sha256_file(order_manifest_path),
        "runtime_readiness_path": str(readiness_evidence_path).replace("\\", "/"),
        "runtime_readiness_sha256": sha256_file(readiness_evidence_path),
        "checks": checks,
        "raw_episode_keys": raw_keys,
        "ordered_episode_keys": ordered_keys,
        "ordered_episode_ids": ordered_ids,
    }


def _exact_episode_coverage(
    dgx_root: Path,
    lane: str,
    expected_episode_keys: list[str],
    order_contract: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"{lane}::"
    runtime_ids = list(order_contract["ordered_episode_ids"])
    expected = [prefix + runtime_id for runtime_id in runtime_ids]
    client = _episode_order(dgx_root / "client" / "client_records.jsonl", prefix)
    controller = _episode_order(
        dgx_root / "onboard" / "controller_records.jsonl",
        prefix,
        allow_prebind_state_only=True,
    )
    checks = {
        "manifest_has_exact_unique_five": len(expected) == 5
        and len(set(expected_episode_keys)) == 5
        and len(set(runtime_ids)) == 5
        and all(runtime_ids),
        "client_exact_order": client["episode_transition_order"] == expected,
        "controller_exact_order": controller["episode_transition_order"] == expected,
        "client_each_episode_observed": set(client["episode_row_counts"]) == set(expected)
        and all(client["episode_row_counts"].get(item, 0) >= 1 for item in expected),
        "controller_each_episode_observed": set(controller["episode_row_counts"])
        == set(expected)
        and all(controller["episode_row_counts"].get(item, 0) >= 1 for item in expected),
        "client_no_unprefixed_episode": client[
            "malformed_or_unprefixed_episode_count"
        ]
        == 0,
        "controller_no_unprefixed_episode": controller[
            "malformed_or_unprefixed_episode_count"
        ]
        == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_prefixed_episode_ids": expected,
        "frozen_trajectory_episode_keys": expected_episode_keys,
        "derived_runtime_episode_ids": runtime_ids,
        "episode_order_contract": order_contract,
        "client": client,
        "controller": controller,
    }


def finalize_episode_coverage(
    *,
    dgx_root: Path,
    lane: str,
    manifest_path: Path,
    order_manifest_path: Path,
    readiness_evidence_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Write a standalone fixed-five receipt for D0.1/D0.2 admission."""

    if lane not in {"a", "b"}:
        raise ContractError("fixed-five coverage lane must be a or b")
    manifest = load_json(manifest_path)
    fixed = manifest.get("fixed_input")
    expected = fixed.get("episode_keys") if isinstance(fixed, dict) else None
    if expected is None:
        expected = manifest.get("episode_keys")
    if not isinstance(expected, list) or not all(
        isinstance(item, str) and item for item in expected
    ):
        raise ContractError("fixed-five coverage manifest has invalid episode keys")
    expected_dataset_sha256 = (
        fixed.get("dataset_file_sha256") if isinstance(fixed, dict) else None
    )
    if expected_dataset_sha256 is None:
        expected_dataset_sha256 = manifest.get("dataset_file_sha256")
    order_contract = _ordered_episode_contract(
        order_manifest_path,
        readiness_evidence_path,
        expected,
        expected_dataset_sha256,
    )
    payload = _exact_episode_coverage(dgx_root, lane, expected, order_contract)
    payload.update(
        {
            "schema_version": 1,
            "lane": lane,
            "dgx_root": str(dgx_root).replace("\\", "/"),
            "recorded_unix": time.time(),
        }
    )
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if payload["status"] != "PASS":
        raise ContractError(f"fixed-five episode coverage failed: {payload['checks']}")
    return payload


def _shared_prep_identity(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "prep_grant_id": binding.get("prep_grant_id"),
        "code_ref_sha": binding.get("code_ref_sha"),
        "golden_bundle_canonical_sha256": binding.get(
            "golden_bundle_canonical_sha256"
        ),
        "run_manifest_canonical_sha256": binding.get(
            "run_manifest_canonical_sha256"
        ),
        "dataset_root": binding.get("dataset_root"),
        "dataset_file_sha256": binding.get("dataset_file_sha256"),
        "static_map_manifest_sha256": binding.get("static_map_manifest_sha256"),
        "source_receipts": binding.get("source_receipts"),
    }


def _single_lane_performance_hashes_bound(
    result_root: Path, runtime: dict[str, Any]
) -> bool:
    recorded = runtime.get("key_file_sha256")
    if not isinstance(recorded, dict):
        return False
    required = {
        "remote/dgx/client/client_summary.json": result_root
        / "remote"
        / "dgx"
        / "client"
        / "client_summary.json",
        "remote/dgx/onboard/controller_summary.json": result_root
        / "remote"
        / "dgx"
        / "onboard"
        / "controller_summary.json",
        "remote/dgx/client/client_records.jsonl": result_root
        / "remote"
        / "dgx"
        / "client"
        / "client_records.jsonl",
        "remote/dgx/onboard/controller_records.jsonl": result_root
        / "remote"
        / "dgx"
        / "onboard"
        / "controller_records.jsonl",
    }
    return all(path.is_file() and recorded.get(name) == sha256_file(path) for name, path in required.items())


def validate_predecessors(
    *,
    prep_root: Path,
    d01_root: Path,
    d02_root: Path,
    grant: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    prep_final_path = prep_root / "d0_prepare_final_summary.json"
    d01_final_path = d01_root / "d0_lane_summary.json"
    d02_final_path = d02_root / "d0_lane_summary.json"
    prep_final = load_json(prep_final_path)
    d01_final = load_json(d01_final_path)
    d02_final = load_json(d02_final_path)
    d01_runtime = _nested_runtime(d01_final, "a")
    d02_runtime = _nested_runtime(d02_final, "b")
    d01_prep = load_json(d01_root / "prep_binding.json")
    d02_prep = load_json(d02_root / "prep_binding.json")
    d01_predecessor = load_json(d01_root / "predecessor_binding.json")
    d02_predecessor = load_json(d02_root / "predecessor_binding.json")
    d01_chain = load_json(d01_root / "preparation_chain_binding.json")
    d02_chain = load_json(d02_root / "preparation_chain_binding.json")

    g = grant
    common = _shared_prep_identity(d01_prep)
    episode_keys = manifest.get("fixed_input", {}).get("episode_keys")
    dataset_sha256 = manifest.get("fixed_input", {}).get("dataset_file_sha256")
    if not isinstance(episode_keys, list) or not all(
        isinstance(item, str) and item for item in episode_keys
    ) or not isinstance(dataset_sha256, str):
        raise ContractError("D0 manifest has invalid frozen episode keys")
    d01_order_contract = _ordered_episode_contract(
        d01_root / "remote" / "x86" / "ordered_episode_manifest.json",
        d01_root
        / "remote"
        / "x86"
        / "health"
        / "runtime_readiness_evidence.json",
        episode_keys,
        dataset_sha256,
    )
    d02_order_contract = _ordered_episode_contract(
        d02_root / "remote" / "x86" / "ordered_episode_manifest.json",
        d02_root
        / "remote"
        / "x86"
        / "health"
        / "runtime_readiness_evidence.json",
        episode_keys,
        dataset_sha256,
    )
    d01_episode_coverage = _exact_episode_coverage(
        d01_root / "remote" / "dgx", "a", episode_keys, d01_order_contract
    )
    d02_episode_coverage = _exact_episode_coverage(
        d02_root / "remote" / "dgx", "b", episode_keys, d02_order_contract
    )
    checks = {
        "prep_final_pass": receipt_passes(prep_final),
        "d01_final_pass": receipt_passes(d01_final),
        "d02_final_pass": receipt_passes(d02_final),
        "d01_runtime_pass": d01_runtime.get("status") == "PASS",
        "d02_runtime_pass": d02_runtime.get("status") == "PASS",
        "d01_expected_stage_lane": d01_runtime.get("stage")
        == "d0_1_lane_a_fixed_5"
        and d01_runtime.get("lane") == "a",
        "d02_expected_stage_lane": d02_runtime.get("stage")
        == "d0_2_lane_b_same_fixed_5"
        and d02_runtime.get("lane") == "b",
        "grant_binds_d02_receipt": g.get("predecessor_stage")
        == "d0_2_lane_b_same_fixed_5"
        and g.get("predecessor_result_root") == str(d02_root).replace("\\", "/")
        and g.get("predecessor_receipt_sha256") == sha256_file(d02_final_path),
        "d02_binds_exact_d01_receipt": d02_predecessor.get("status") == "PASS"
        and d02_predecessor.get("predecessor_stage") == "d0_1_lane_a_fixed_5"
        and d02_predecessor.get("predecessor_result_root")
        == str(d01_root).replace("\\", "/")
        and d02_predecessor.get("predecessor_receipt_sha256")
        == sha256_file(d01_final_path),
        "d01_internal_receipt_hashes": d01_runtime.get("prep_binding_sha256")
        == sha256_file(d01_root / "prep_binding.json")
        and d01_runtime.get("predecessor_binding_sha256")
        == sha256_file(d01_root / "predecessor_binding.json")
        and d01_runtime.get("preparation_chain_binding_sha256")
        == sha256_file(d01_root / "preparation_chain_binding.json"),
        "d02_internal_receipt_hashes": d02_runtime.get("prep_binding_sha256")
        == sha256_file(d02_root / "prep_binding.json")
        and d02_runtime.get("predecessor_binding_sha256")
        == sha256_file(d02_root / "predecessor_binding.json")
        and d02_runtime.get("preparation_chain_binding_sha256")
        == sha256_file(d02_root / "preparation_chain_binding.json"),
        "both_predecessor_bindings_pass": receipt_passes(d01_predecessor)
        and receipt_passes(d02_predecessor),
        "same_d0_0_preparation": _shared_prep_identity(d02_prep) == common,
        "d01_preparation_chain_pass": receipt_passes(d01_chain),
        "d02_preparation_chain_pass": receipt_passes(d02_chain),
        "same_code_ref": d01_runtime.get("code_ref_sha")
        == d02_runtime.get("code_ref_sha")
        == g.get("code_ref_sha"),
        "same_golden": d01_runtime.get("golden_bundle_canonical_sha256")
        == d02_runtime.get("golden_bundle_canonical_sha256")
        == g.get("golden_bundle_sha256"),
        "same_run_manifest": d01_runtime.get("run_manifest_canonical_sha256")
        == d02_runtime.get("run_manifest_canonical_sha256")
        == g.get("run_manifest_sha256"),
        "same_fixed_five_count": d01_runtime.get("episode_count") == 5
        and d02_runtime.get("episode_count") == 5,
        "single_lane_performance_hashes_bound": _single_lane_performance_hashes_bound(
            d01_root, d01_runtime
        )
        and _single_lane_performance_hashes_bound(d02_root, d02_runtime),
        "single_lane_exact_episode_coverage": d01_episode_coverage["status"]
        == "PASS"
        and d02_episode_coverage["status"] == "PASS",
        "distinct_lane_deployment_roots": d01_prep.get("deployment_roots")
        != d02_prep.get("deployment_roots"),
    }
    prep_summary = load_json(prep_root / "d0_prepare_summary.json")
    prep_release_path = prep_root / "lease_release_summary.json"
    prep_dataset_path = prep_root / "remote" / "x86" / "dataset_audit.json"
    prep_release = load_json(prep_release_path)
    prep_dataset = load_json(prep_dataset_path)
    expected_source_receipts = {
        "prepare_summary_sha256": sha256_file(prep_root / "d0_prepare_summary.json"),
        "prepare_release_sha256": sha256_file(prep_release_path),
        "dataset_audit_sha256": sha256_file(prep_dataset_path),
        "prepare_final_sha256": sha256_file(prep_final_path),
    }
    roots = prep_summary.get("deployment_roots")
    if not isinstance(roots, dict):
        roots = {}
    checks.update(
        {
            "prep_summary_pass": receipt_passes(prep_summary),
            "prep_release_pass": receipt_passes(prep_release),
            "prep_dataset_pass": prep_dataset.get("status") == "PASS",
            "exact_d00_source_receipts": common.get("source_receipts")
            == expected_source_receipts,
            "prep_grant_matches": prep_summary.get("grant_id")
            == common.get("prep_grant_id"),
            "all_four_deployment_roots": all(
                isinstance(roots.get(name), str) and roots.get(name)
                for name in ("dgx_a", "dgx_b", "x86_a", "x86_b")
            ),
            "lane_binding_roots_match_prep": d01_prep.get("deployment_roots")
            == {"dgx": roots.get("dgx_a"), "x86": roots.get("x86_a")}
            and d02_prep.get("deployment_roots")
            == {"dgx": roots.get("dgx_b"), "x86": roots.get("x86_b")},
        }
    )
    memory = prep_summary.get("memory_admission")
    if not isinstance(memory, dict):
        memory = {}
    minimum_memory = memory.get("minimum_available_bytes")
    checks["dual_memory_contract_present"] = (
        isinstance(minimum_memory, int) and minimum_memory > 0
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "grant_id": g.get("grant_id"),
        "authorization_ref_sha": g.get("authorization_ref_sha"),
        "code_ref_sha": g.get("code_ref_sha"),
        "golden_bundle_canonical_sha256": g.get("golden_bundle_sha256"),
        "run_manifest_canonical_sha256": g.get("run_manifest_sha256"),
        "prep_grant_id": common.get("prep_grant_id"),
        "deployment_roots": roots,
        "dataset_root": common.get("dataset_root"),
        "dataset_file_sha256": common.get("dataset_file_sha256"),
        "static_map_manifest_sha256": common.get("static_map_manifest_sha256"),
        "minimum_available_memory_bytes": minimum_memory,
        "single_lane_baseline": {
            "a": {
                "result_root": str(d01_root).replace("\\", "/"),
                "receipt_sha256": sha256_file(d01_final_path),
                "runtime_duration_sec": _duration_from_tree(
                    d01_root / "remote" / "x86"
                ),
                "performance_signals": _performance_signals(
                    d01_root / "remote" / "dgx", d01_root / "remote" / "x86"
                ),
                "episode_coverage": d01_episode_coverage,
            },
            "b": {
                "result_root": str(d02_root).replace("\\", "/"),
                "receipt_sha256": sha256_file(d02_final_path),
                "runtime_duration_sec": _duration_from_tree(
                    d02_root / "remote" / "x86"
                ),
                "performance_signals": _performance_signals(
                    d02_root / "remote" / "dgx", d02_root / "remote" / "x86"
                ),
                "episode_coverage": d02_episode_coverage,
            },
        },
        "source_receipts": {
            "prep_final_sha256": sha256_file(prep_final_path),
            "d01_final_sha256": sha256_file(d01_final_path),
            "d02_final_sha256": sha256_file(d02_final_path),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise ContractError(f"D0.3 predecessor chain failed: {checks}")
    return payload


def _find_attempt(x86_root: Path, lane: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = sorted(
        (x86_root / "evaluator").glob(
            f"t5_lane_{lane}_model*_model_attempt_*/result.json"
        )
    )
    if len(candidates) != 1:
        raise ContractError(
            f"Lane {lane} needs exactly one fixed-five result, found {len(candidates)}"
        )
    result_path = candidates[0]
    validation_path = result_path.parent / "isaac_remote_validation.json"
    return result_path.parent, load_json(result_path), load_json(validation_path)


def _finite_metrics(result: dict[str, Any]) -> tuple[dict[str, float], int]:
    metrics = result.get("val_unseen", result)
    if not isinstance(metrics, dict):
        raise ContractError("evaluator metrics are not an object")
    try:
        required = {name: float(metrics[name]) for name in ("SR", "OS", "SPL", "NE")}
        count = int(metrics.get("Count", metrics.get("length")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("missing D0 SR/OS/SPL/NE/Count metrics") from exc
    if not all(math.isfinite(value) for value in required.values()):
        raise ContractError("non-finite D0 metric")
    return required, count


def _contains_prefix(value: Any, prefix: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_prefix(item, prefix) for item in value.values())
    if isinstance(value, list):
        return any(_contains_prefix(item, prefix) for item in value)
    return isinstance(value, str) and value.startswith(prefix)


def _ledger_clean(path: Path) -> bool:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    started = {
        (row.get("scope", "host"), row.get("component"), row.get("pid"))
        for row in rows
        if row.get("event") == "started" and row.get("pid") is not None
    }
    absent = {
        (row.get("scope", "host"), row.get("component"), row.get("pid"))
        for row in rows
        if row.get("event") == "verified_absent" and row.get("pid") is not None
    }
    return bool(started) and started.issubset(absent)


def _lane_evidence(result_root: Path, lane: str, manifest: dict[str, Any]) -> dict[str, Any]:
    base = result_root / "remote" / f"lane_{lane}"
    dgx, x86 = base / "dgx", base / "x86"
    dgx_status = load_json(dgx / "lane_status.json")
    dgx_ready = load_json(dgx / "lane_ready.json")
    model_identity = load_json(dgx / "model_identity_audit.json")
    token_process_audit = load_json(dgx / "hf_token_process_audit.json")
    dgx_contract = load_json(dgx / "lane_contract.json")
    x86_status = load_json(x86 / "isaac_status.json")
    x86_contract = load_json(x86 / "isaac_contract.json")
    gpu_mapping = load_json(x86 / "gpu_mapping.json")
    kit_gpu = load_json(x86 / "kit_gpu_audit.json")
    attempt, result, validation = _find_attempt(x86, lane)
    metrics, count = _finite_metrics(result)
    lane_contract = manifest["lanes"][lane]
    own_prefix, other_prefix = f"{lane}::", "b::" if lane == "a" else "a::"
    duration = _duration_from_tree(x86)
    performance_signals = _performance_signals(dgx, x86)
    identity_scan = _scan_lane_identity(dgx, x86, lane)
    expected_episode_keys = manifest.get("fixed_input", {}).get("episode_keys")
    expected_dataset_sha256 = manifest.get("fixed_input", {}).get(
        "dataset_file_sha256"
    )
    if not isinstance(expected_episode_keys, list) or not all(
        isinstance(item, str) and item for item in expected_episode_keys
    ) or not isinstance(expected_dataset_sha256, str):
        raise ContractError("D0 manifest has invalid frozen episode keys")
    order_contract = _ordered_episode_contract(
        x86 / "ordered_episode_manifest.json",
        x86 / "health" / "runtime_readiness_evidence.json",
        expected_episode_keys,
        expected_dataset_sha256,
    )
    episode_coverage = _exact_episode_coverage(
        dgx, lane, expected_episode_keys, order_contract
    )
    checks = {
        "fixed_five_completed": count == 5
        and validation.get("episode_count") == 5
        and validation.get("expected_episode_count") == 5,
        "evaluator_validation": validation.get("status") == "PASS",
        "metrics_ranges": 0.0 <= metrics["SR"] <= 1.0
        and 0.0 <= metrics["OS"] <= 1.0
        and 0.0 <= metrics["SPL"] <= 1.0
        and metrics["NE"] >= 0.0,
        "sr_is_report_only": float(validation.get("minimum_success_rate", -1.0))
        == 0.0,
        "dgx_cleanup": dgx_status.get("status") == "PASS"
        and dgx_status.get("residual_count") == 0,
        "dgx_ready": dgx_ready.get("status") == "READY"
        and dgx_ready.get("lane") == lane
        and dgx_ready.get("mode") == "model",
        "x86_cleanup": x86_status.get("status") == "PASS"
        and x86_status.get("residual_count") == 0
        and x86_status.get("socket_residual_count") == 0
        and x86_status.get("clock_publishers_after_stop") == 0
        and x86_status.get("lane_lock_released") is True,
        "fixed_dataset_execution": x86_status.get("execution_profile")
        == "fixed_dataset"
        and x86_status.get("episode_acceptance_claimed") is True
        and x86_status.get("evaluation_completed_naturally") is True
        and x86_status.get("termination_reason") == "evaluator_natural_exit",
        "model_identity": model_identity.get("status") == "PASS"
        and bool(model_identity.get("checks"))
        and all(item is True for item in model_identity.get("checks", {}).values()),
        "credential_process_scope": token_process_audit.get("status") == "PASS"
        and token_process_audit.get("exact_secret_match_count", 0) >= 1
        and token_process_audit.get("allowed_model_match_count")
        == token_process_audit.get("exact_secret_match_count")
        and token_process_audit.get("disallowed_match_count") == 0
        and token_process_audit.get("parent_match_count") == 0
        and token_process_audit.get("onboard_match_count") == 0
        and token_process_audit.get("evaluator_match_count") == 0
        and token_process_audit.get("secret_value_recorded") is False
        and token_process_audit.get("secret_digest_recorded") is False
        and bool(token_process_audit.get("checks"))
        and all(
            item is True for item in token_process_audit.get("checks", {}).values()
        ),
        "lane_identity": dgx_contract.get("lane") == lane
        and x86_contract.get("lane") == lane
        and dgx_contract.get("identity_prefix") == own_prefix
        and x86_contract.get("identity", {}).get("episode_prefix") == own_prefix,
        "dds_namespace": dgx_contract.get("ros_domain_id")
        == x86_contract.get("ros_domain_id")
        == int(lane_contract["ros_domain_id"])
        and dgx_contract.get("namespace") == lane_contract["namespace"]
        and x86_contract.get("namespace") == lane_contract["namespace"],
        "ports": x86_contract.get("ports")
        == {
            "controller": int(lane_contract["controller_port"]),
            "model_client": int(lane_contract["model_client_port"]),
            "oracle": int(lane_contract["oracle_port"]),
            "clock": int(lane_contract["clock_port"]),
        },
        "gpu_and_cpuset": gpu_mapping.get("status") == "PASS"
        and kit_gpu.get("status") == "PASS"
        and gpu_mapping.get("host_physical_gpu_index")
        == int(lane_contract["isaac_render_gpu_physical"])
        and x86_contract.get("cpuset") == lane_contract["cpuset"],
        "parallel_output_policy": x86_contract.get("full_mp4_encoding_allowed")
        is False
        and x86_contract.get("video_policy") == "jsonl_keyframes_only"
        and x86_contract.get("shared_asset_lock_mode") == "shared_read",
        "no_other_lane_identity": identity_scan["opposite_prefix_match_count"] == 0
        and not _contains_prefix(result, other_prefix)
        and not _contains_prefix(validation, other_prefix),
        "own_real_request_reset_episode_identity": identity_scan[
            "client_record_own_prefix_match_count"
        ]
        >= 1
        and identity_scan["sensor_record_own_prefix_match_count"] >= 1,
        "exact_frozen_episode_coverage": episode_coverage["status"] == "PASS"
        and all(item is True for item in episode_coverage["checks"].values()),
        "frozen_dataset_hash": x86_contract.get("dataset", {}).get("sha256")
        == manifest.get("fixed_input", {}).get("dataset_file_sha256"),
        "pid_ledgers_clean": _ledger_clean(dgx / "pid_ledger.jsonl")
        and _ledger_clean(x86 / "pid_ledger.jsonl"),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
        "episode_count": count,
        "runtime_duration_sec": duration,
        "performance_signals": performance_signals,
        "identity_scan": identity_scan,
        "episode_coverage": episode_coverage,
        "started_unix": float(x86_contract["started_unix"]),
        "finished_unix": float(x86_status["finished_unix"]),
        "attempt_relative": str(attempt.relative_to(result_root)).replace("\\", "/"),
        "cache_roots": x86_contract.get("cache_roots"),
        "health_endpoint": x86_contract.get("health_endpoint"),
        "ipc_alias": x86_contract.get("ipc_alias"),
        "identity_prefix": own_prefix,
        "golden_bundle_canonical_sha256": model_identity.get(
            "golden_bundle_canonical_sha256"
        ),
        "ready_golden_bundle_canonical_sha256": dgx_ready.get(
            "golden_bundle_canonical_sha256"
        ),
        "static_map_manifest_sha256": dgx_contract.get("inputs", {})
        .get("static_map_manifest", {})
        .get("sha256"),
        "other_identity_prefix_observation_count": 0
        if checks["no_other_lane_identity"]
        else None,
    }


def finalize_dual(
    *,
    result_root: Path,
    predecessor_binding: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    predecessor = load_json(predecessor_binding)
    manifest = load_json(manifest_path)
    cleanup = load_json(result_root / "audits" / "coordinator_cleanup_receipt.json")
    isolation = load_json(result_root / "audits" / "cross_lane_isolation.json")
    capacity = load_json(result_root / "audits" / "capacity_prestart.json")
    credential = load_json(result_root / "audits" / "credential_exposure_audit.json")
    lanes = {
        lane: _lane_evidence(result_root, lane, manifest) for lane in ("a", "b")
    }
    baseline = predecessor.get("single_lane_baseline", {})
    maximum = float(manifest["acceptance"]["maximum_simultaneous_slowdown_fraction"])
    if not math.isclose(maximum, 0.2, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError("D0.3 slowdown threshold must remain frozen at 0.2")
    performance: dict[str, Any] = {}
    for lane in ("a", "b"):
        base_duration = float(baseline[lane]["runtime_duration_sec"])
        dual_duration = float(lanes[lane]["runtime_duration_sec"])
        duration_slowdown = dual_duration / base_duration - 1.0
        base_signals = baseline[lane]["performance_signals"]
        dual_signals = lanes[lane]["performance_signals"]
        dimensions: dict[str, Any] = {
            "runtime_duration_sec": {
                "direction": "lower_is_better",
                "single_lane": base_duration,
                "simultaneous": dual_duration,
                "degradation_fraction": duration_slowdown,
                "status": "PASS" if duration_slowdown <= maximum else "FAIL",
            }
        }
        for name in _LOWER_IS_BETTER:
            base_value = _positive_finite(
                base_signals["lower_is_better"].get(name), name
            )
            dual_value = _positive_finite(
                dual_signals["lower_is_better"].get(name), name
            )
            degradation = dual_value / base_value - 1.0
            dimensions[name] = {
                "direction": "lower_is_better",
                "single_lane": base_value,
                "simultaneous": dual_value,
                "degradation_fraction": degradation,
                "status": "PASS" if degradation <= maximum else "FAIL",
            }
        base_hz = _positive_finite(
            base_signals["higher_is_better"].get("measured_control_hz"),
            "measured_control_hz",
        )
        dual_hz = _positive_finite(
            dual_signals["higher_is_better"].get("measured_control_hz"),
            "measured_control_hz",
        )
        throughput_drop = 1.0 - dual_hz / base_hz
        dimensions["measured_control_hz"] = {
            "direction": "higher_is_better",
            "single_lane": base_hz,
            "simultaneous": dual_hz,
            "degradation_fraction": throughput_drop,
            "status": "PASS" if throughput_drop <= maximum else "FAIL",
        }
        performance[lane] = {
            "single_lane_duration_sec": base_duration,
            "simultaneous_duration_sec": dual_duration,
            "slowdown_fraction": duration_slowdown,
            "maximum_allowed_slowdown_fraction": maximum,
            "dimensions": dimensions,
            "sources": dual_signals.get("sources"),
            "status": "PASS"
            if all(value["status"] == "PASS" for value in dimensions.values())
            else "FAIL",
        }
    starts = [lanes[lane]["started_unix"] for lane in ("a", "b")]
    finishes = [lanes[lane]["finished_unix"] for lane in ("a", "b")]
    overlap = max(0.0, min(finishes) - max(starts))
    shorter = min(lanes[lane]["runtime_duration_sec"] for lane in ("a", "b"))
    overlap_fraction = overlap / shorter if shorter > 0 else 0.0
    cache_a, cache_b = lanes["a"].get("cache_roots"), lanes["b"].get("cache_roots")
    cache_values_a = set(cache_a.values()) if isinstance(cache_a, dict) else set()
    cache_values_b = set(cache_b.values()) if isinstance(cache_b, dict) else set()
    expected_keys = manifest.get("fixed_input", {}).get("episode_keys")
    checks = {
        "predecessor_binding_pass": receipt_passes(predecessor),
        "both_lanes_pass": all(lanes[lane]["status"] == "PASS" for lane in lanes),
        "both_lanes_completed_five": all(
            lanes[lane]["episode_count"] == 5 for lane in lanes
        ),
        "both_lanes_exact_golden": all(
            lanes[lane].get("golden_bundle_canonical_sha256")
            == lanes[lane].get("ready_golden_bundle_canonical_sha256")
            == predecessor.get("golden_bundle_canonical_sha256")
            for lane in lanes
        ),
        "both_lanes_exact_static_map": all(
            lanes[lane].get("static_map_manifest_sha256")
            == predecessor.get("static_map_manifest_sha256")
            for lane in lanes
        ),
        "same_frozen_five": isinstance(expected_keys, list)
        and len(expected_keys) == 5
        and manifest.get("fixed_input", {}).get("same_episode_on_two_lanes_counts_once")
        is True,
        "simultaneous_execution": overlap > 0.0 and overlap_fraction >= 0.8,
        "slowdown_within_twenty_percent": all(
            performance[lane]["status"] == "PASS" for lane in performance
        ),
        "all_required_performance_dimensions_present": all(
            set(performance[lane]["dimensions"])
            == {
                "runtime_duration_sec",
                "mean_inference_latency_sec",
                "mean_action_round_trip_latency_sec",
                "mean_nav2_resolution_latency_sec",
                "measured_control_hz",
            }
            for lane in performance
        ),
        "capacity_recheck_pass": receipt_passes(capacity),
        "cross_lane_isolation_pass": receipt_passes(isolation),
        "cleanup_pass": receipt_passes(cleanup),
        "credential_not_exposed": credential.get("status") == "PASS"
        and credential.get("exact_secret_match_count") == 0,
        "cache_roots_disjoint": bool(cache_values_a)
        and bool(cache_values_b)
        and cache_values_a.isdisjoint(cache_values_b),
        "health_endpoints_disjoint": lanes["a"].get("health_endpoint")
        != lanes["b"].get("health_endpoint"),
        "ipc_aliases_disjoint": lanes["a"].get("ipc_alias")
        != lanes["b"].get("ipc_alias"),
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "d0_3_dual_lane_same_fixed_5",
        "grant_id": predecessor.get("grant_id"),
        "authorization_ref_sha": predecessor.get("authorization_ref_sha"),
        "code_ref_sha": predecessor.get("code_ref_sha"),
        "golden_bundle_canonical_sha256": predecessor.get(
            "golden_bundle_canonical_sha256"
        ),
        "run_manifest_canonical_sha256": predecessor.get(
            "run_manifest_canonical_sha256"
        ),
        "checks": checks,
        "lanes": lanes,
        "performance": performance,
        "simultaneous_window": {
            "overlap_sec": overlap,
            "overlap_fraction_of_shorter_lane": overlap_fraction,
            "required_minimum_fraction": 0.8,
        },
        "paired_statistics": {
            "frozen_episode_keys": expected_keys,
            "unique_statistical_sample_count": 5,
            "hardware_reproduction_observation_count": 10,
            "same_episode_results_count_once": True,
            "metrics_are_per_lane": True,
            "pooled_sr_os_spl_ne_forbidden": True,
            "held_out_tuning_used": False,
            "comparison_role": "frozen_hardware_reproduction_and_capacity_gate",
        },
        "fallback": {
            "required": not checks["slowdown_within_twenty_percent"],
            "action": "INTERLEAVED_SINGLE_LANE_EXECUTION"
            if not checks["slowdown_within_twenty_percent"]
            else None,
            "preserve_failed_parallel_evidence": True,
        },
        "evidence_sha256": {
            "predecessor_binding": sha256_file(predecessor_binding),
            "capacity_prestart": sha256_file(
                result_root / "audits" / "capacity_prestart.json"
            ),
            "cross_lane_isolation": sha256_file(
                result_root / "audits" / "cross_lane_isolation.json"
            ),
            "coordinator_cleanup": sha256_file(
                result_root / "audits" / "coordinator_cleanup_receipt.json"
            ),
        },
        "recorded_unix": time.time(),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise ContractError(f"D0.3 finalization failed: {checks}")
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    previous = commands.add_parser("predecessors")
    previous.add_argument("--prep-root", type=Path, required=True)
    previous.add_argument("--d01-root", type=Path, required=True)
    previous.add_argument("--d02-root", type=Path, required=True)
    previous.add_argument("--grant", type=Path, required=True)
    previous.add_argument("--manifest", type=Path, required=True)
    previous.add_argument("--output", type=Path, required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--result-root", type=Path, required=True)
    final.add_argument("--predecessor-binding", type=Path, required=True)
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    episodes = commands.add_parser("episodes")
    episodes.add_argument("--dgx-root", type=Path, required=True)
    episodes.add_argument("--lane", choices=("a", "b"), required=True)
    episodes.add_argument("--manifest", type=Path, required=True)
    episodes.add_argument("--order-manifest", type=Path, required=True)
    episodes.add_argument("--readiness-evidence", type=Path, required=True)
    episodes.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "predecessors":
            validate_predecessors(
                prep_root=args.prep_root,
                d01_root=args.d01_root,
                d02_root=args.d02_root,
                grant=load_json(args.grant).get("grant", load_json(args.grant)),
                manifest=load_json(args.manifest),
                output=args.output,
            )
        elif args.command == "finalize":
            finalize_dual(
                result_root=args.result_root,
                predecessor_binding=args.predecessor_binding,
                manifest_path=args.manifest,
                output=args.output,
            )
        else:
            finalize_episode_coverage(
                dgx_root=args.dgx_root,
                lane=args.lane,
                manifest_path=args.manifest,
                order_manifest_path=args.order_manifest,
                readiness_evidence_path=args.readiness_evidence,
                output=args.output,
            )
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
