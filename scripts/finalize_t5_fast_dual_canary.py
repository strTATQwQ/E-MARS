#!/usr/bin/env python3
"""Finalize the two independent T5 fast engineering canaries.

This is deliberately a concurrency/isolation receipt, not an episode result.
It proves that Lane A and Lane B each produced a valid 60-second engineering
canary and were simultaneously RUNNING for at least 45 seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MINIMUM_OVERLAP_SECONDS = 45.0
EXPECTED_CONFIGURED_SECONDS = 60
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_CANARY_CHECKS = {
    "bounded_duration_valid",
    "ready_interval_complete",
    "ready_probe_pass",
    "ready_baselines_valid",
    "cutoff_values_valid",
    "cutoff_counter_not_rolled_back",
    "sample_baselines_match_ready",
    "periodic_sample_count",
    "timeline_starts_near_zero",
    "timeline_strictly_increasing",
    "timeline_covers_configured_window",
    "timeline_sample_gap_bounded",
    "health_running_every_sample",
    "model_action_counts_valid",
    "model_action_count_positive_at_ready",
    "model_action_counts_monotonic",
    "model_actions_advance",
    "model_step_error_counts_valid",
    "model_step_error_counts_monotonic_from_ready",
    "new_model_step_error_counts_valid",
    "model_step_errors_absent",
    "new_model_step_errors_absent",
    "fresh_five_episode_evaluator_state",
    "model_action_count_not_rolled_back_at_tail",
    "model_step_error_count_not_rolled_back_at_tail",
    "evaluator_total_path_stable_at_tail",
    "clock_state_age_seconds_le_15",
    "clock_counters_monotonic",
    "physics_steps_advance",
    "physics_not_stagnant_over_30_seconds",
    "evaluator_stop_completed_before_analysis",
}
EXPECTED_LANE_ISOLATION = {
    "a": {
        "gpu": 0,
        "ros_domain_id": 75,
        "namespace": "/t5/lane_a",
        "cpuset": "0,2,4,6,8,10,12,14,16",
        "ports": {
            "controller": 25137,
            "model_client": 25139,
            "oracle": 25140,
            "clock": 25141,
        },
    },
    "b": {
        "gpu": 1,
        "ros_domain_id": 76,
        "namespace": "/t5/lane_b",
        "cpuset": "1,3,5,7,9,11,13,15,17",
        "ports": {
            "controller": 25138,
            "model_client": 25239,
            "oracle": 25240,
            "clock": 25241,
        },
    },
}


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if path.is_symlink():
        return None, "symlink is not accepted"
    if not path.is_file():
        return None, "regular file is missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    return value, None


def _checks_pass(value: Any, required: Optional[set] = None) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if required is not None and not required.issubset(value):
        return False
    return all(item is True for item in value.values())


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _parse_cpuset(value: Any) -> Optional[set]:
    if not isinstance(value, str) or not value:
        return None
    cpus = set()
    try:
        for fragment in value.split(","):
            bounds = fragment.split("-")
            if len(bounds) == 1:
                start = finish = int(bounds[0])
            elif len(bounds) == 2:
                start, finish = int(bounds[0]), int(bounds[1])
            else:
                return None
            if start < 0 or finish < start:
                return None
            cpus.update(range(start, finish + 1))
    except ValueError:
        return None
    return cpus if cpus else None


def _binding_identity(binding: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(binding, dict):
        return None
    receipts = binding.get("source_receipts")
    if not isinstance(receipts, dict) or set(receipts) != {
        "prepare_summary_sha256",
        "prepare_final_sha256",
        "fast_prepare_input_sha256",
        "dataset_audit_sha256",
    }:
        return None
    if not all(SHA256.fullmatch(str(value)) for value in receipts.values()):
        return None
    dataset_sha = binding.get("dataset_sha256")
    map_sha = binding.get("static_map_manifest_sha256")
    episode_keys = binding.get("episode_keys")
    if (
        SHA256.fullmatch(str(dataset_sha)) is None
        or SHA256.fullmatch(str(map_sha)) is None
        or binding.get("episode_count") != 5
        or not isinstance(episode_keys, list)
        or len(episode_keys) != 5
        or any(not isinstance(item, str) or not item for item in episode_keys)
    ):
        return None
    return {
        "source_receipts": receipts,
        "dataset_sha256": dataset_sha,
        "static_map_manifest_sha256": map_sha,
        "episode_count": 5,
        "episode_keys": episode_keys,
    }


def _lane_evidence(result_root: Path, expected_lane: str) -> Dict[str, Any]:
    paths = {
        "final": result_root / "fast_lane_final_summary.json",
        "runtime": result_root / "fast_lane_summary.json",
        "canary": result_root / "remote" / "x86" / "engineering_canary.json",
    }
    documents: Dict[str, Optional[Dict[str, Any]]] = {}
    errors: Dict[str, str] = {}
    hashes: Dict[str, Optional[str]] = {}
    for name, path in paths.items():
        document, error = _load_object(path)
        documents[name] = document
        hashes[name] = _sha256(path)
        if error is not None:
            errors[name] = error

    final = documents["final"] or {}
    runtime = documents["runtime"] or {}
    canary = documents["canary"] or {}
    binding = runtime.get("input_binding")
    identity = _binding_identity(binding)
    x86_status = runtime.get("x86_status")
    x86_status = x86_status if isinstance(x86_status, dict) else {}
    sealed_evidence = runtime.get("sealed_evidence")
    sealed_evidence = sealed_evidence if isinstance(sealed_evidence, dict) else {}
    canary_seal = sealed_evidence.get("engineering_canary")
    canary_seal = canary_seal if isinstance(canary_seal, dict) else {}
    expected_isolation = EXPECTED_LANE_ISOLATION[expected_lane]
    observed_isolation = {
        "gpu": x86_status.get("isaac_render_gpu_physical_index"),
        "cuda_visible_devices": x86_status.get("cuda_visible_devices"),
        "host_cuda_visible_devices": x86_status.get("host_cuda_visible_devices"),
        "container_logical_cuda_visible_devices": x86_status.get(
            "container_logical_cuda_visible_devices"
        ),
        "isaac_physics_gpu_visible_index": x86_status.get(
            "isaac_physics_gpu_visible_index"
        ),
        "ros_domain_id": x86_status.get("ros_domain_id"),
        "namespace": x86_status.get("namespace"),
        "cpuset": x86_status.get("cpuset"),
        "ports": x86_status.get("ports"),
    }
    started = _finite_number(canary.get("started_unix"))
    finished = _finite_number(canary.get("finished_unix"))
    configured = canary.get("configured_seconds")
    observed = _finite_number(canary.get("observed_seconds"))

    checks = {
        "all_source_files_regular": not errors and all(hashes.values()),
        "final_summary_pass": final.get("status") == "PASS"
        and _checks_pass(final.get("checks")),
        "runtime_summary_pass": runtime.get("status") == "PASS"
        and _checks_pass(runtime.get("checks")),
        "final_embeds_exact_runtime_summary": final.get("runtime_summary") == runtime,
        "lane_identity": runtime.get("lane") == expected_lane
        and canary.get("lane") == expected_lane
        and runtime.get("resource_profile") == "lane-%s" % expected_lane,
        "canary_profile": runtime.get("profile") == "canary60"
        and canary.get("profile") == "engineering_canary",
        "canary_pass": canary.get("status") == "PASS"
        and _checks_pass(canary.get("checks"), REQUIRED_CANARY_CHECKS),
        "canary_sha256_matches_sealed_summary": (
            canary_seal.get("applicability") == "required"
            and canary_seal.get("relative_path")
            == "remote/x86/engineering_canary.json"
            and SHA256.fullmatch(str(canary_seal.get("sha256"))) is not None
            and hashes["canary"] == canary_seal.get("sha256")
        ),
        "configured_sixty_seconds": configured == EXPECTED_CONFIGURED_SECONDS
        and observed is not None
        and observed >= EXPECTED_CONFIGURED_SECONDS,
        "no_episode_acceptance_claim": canary.get("episode_acceptance_claimed")
        is False
        and canary.get("evaluation_completed_naturally") is False,
        "ordered_finite_running_window": started is not None
        and finished is not None
        and finished > started,
        "binding_pass": isinstance(binding, dict)
        and binding.get("status") == "PASS"
        and _checks_pass(binding.get("checks"))
        and binding.get("lane") == expected_lane
        and binding.get("code_ref_sha") == runtime.get("code_ref_sha")
        and identity is not None,
        "code_sha_valid": SHA40.fullmatch(str(runtime.get("code_ref_sha")))
        is not None,
        "x86_status_identity": x86_status.get("status") == "PASS"
        and x86_status.get("lane") == expected_lane,
        "fixed_gpu_contract": (
            _exact_int(
                x86_status.get("isaac_render_gpu_physical_index"),
                expected_isolation["gpu"],
            )
            and x86_status.get("cuda_visible_devices")
            == str(expected_isolation["gpu"])
            and x86_status.get("host_cuda_visible_devices")
            == str(expected_isolation["gpu"])
            and x86_status.get("container_logical_cuda_visible_devices") == "0"
            and _exact_int(x86_status.get("isaac_physics_gpu_visible_index"), 0)
        ),
        "fixed_ros_domain_contract": _exact_int(
            x86_status.get("ros_domain_id"), expected_isolation["ros_domain_id"]
        ),
        "fixed_namespace_contract": x86_status.get("namespace")
        == expected_isolation["namespace"],
        "fixed_cpuset_contract": x86_status.get("cpuset")
        == expected_isolation["cpuset"],
        "fixed_ports_contract": x86_status.get("ports")
        == expected_isolation["ports"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "lane": expected_lane,
        "result_root": str(result_root),
        "code_ref_sha": runtime.get("code_ref_sha"),
        "preparation_binding": identity,
        "sealed_engineering_canary": canary_seal,
        "isolation_contract": {
            "expected": expected_isolation,
            "observed": observed_isolation,
        },
        "running_window": {
            "started_unix": started,
            "finished_unix": finished,
            "duration_seconds": finished - started
            if started is not None and finished is not None
            else None,
        },
        "checks": checks,
        "source_errors": errors,
        "evidence_sha256": hashes,
    }


def finalize(lane_a_root: Path, lane_b_root: Path) -> Dict[str, Any]:
    lanes = {
        "a": _lane_evidence(lane_a_root, "a"),
        "b": _lane_evidence(lane_b_root, "b"),
    }
    starts = [lanes[lane]["running_window"]["started_unix"] for lane in ("a", "b")]
    finishes = [
        lanes[lane]["running_window"]["finished_unix"] for lane in ("a", "b")
    ]
    timestamps_valid = all(value is not None for value in starts + finishes)
    overlap_start = max(starts) if timestamps_valid else None
    overlap_finish = min(finishes) if timestamps_valid else None
    overlap_seconds = (
        max(0.0, overlap_finish - overlap_start)
        if overlap_start is not None and overlap_finish is not None
        else 0.0
    )
    code_sha = lanes["a"].get("code_ref_sha")
    binding = lanes["a"].get("preparation_binding")
    isolation_a = lanes["a"]["isolation_contract"]["observed"]
    isolation_b = lanes["b"]["isolation_contract"]["observed"]
    gpu_values = [isolation_a.get("gpu"), isolation_b.get("gpu")]
    domain_values = [
        isolation_a.get("ros_domain_id"),
        isolation_b.get("ros_domain_id"),
    ]
    cpuset_values = [
        _parse_cpuset(isolation_a.get("cpuset")),
        _parse_cpuset(isolation_b.get("cpuset")),
    ]
    port_values = []
    for isolation in (isolation_a, isolation_b):
        ports = isolation.get("ports")
        port_values.append(
            set(ports.values())
            if isinstance(ports, dict)
            and ports
            and all(type(value) is int for value in ports.values())
            else None
        )
    checks = {
        "both_lanes_pass": all(lanes[lane]["status"] == "PASS" for lane in lanes),
        "same_code_sha": code_sha is not None
        and code_sha == lanes["b"].get("code_ref_sha"),
        "same_preparation_binding": binding is not None
        and binding == lanes["b"].get("preparation_binding"),
        "running_windows_valid": timestamps_valid,
        "minimum_45_second_overlap": overlap_seconds >= MINIMUM_OVERLAP_SECONDS,
        "episode_acceptance_not_claimed": all(
            lanes[lane]["checks"]["no_episode_acceptance_claim"] is True
            for lane in lanes
        ),
        "gpus_disjoint": all(type(value) is int for value in gpu_values)
        and len(set(gpu_values)) == 2,
        "ros_domains_disjoint": all(type(value) is int for value in domain_values)
        and len(set(domain_values)) == 2,
        "namespaces_disjoint": isinstance(isolation_a.get("namespace"), str)
        and isinstance(isolation_b.get("namespace"), str)
        and isolation_a.get("namespace") != isolation_b.get("namespace"),
        "cpusets_disjoint": all(value is not None for value in cpuset_values)
        and cpuset_values[0].isdisjoint(cpuset_values[1]),
        "ports_disjoint": all(value is not None for value in port_values)
        and port_values[0].isdisjoint(port_values[1]),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "t5_fast_dual_lane_engineering_canary",
        "evidence_role": "dual_lane_concurrency_and_isolation_only",
        "episode_acceptance_claimed": False,
        "evaluation_completed_naturally": False,
        "statistical_sample_count": 0,
        "code_ref_sha": code_sha if checks["same_code_sha"] else None,
        "preparation_binding": binding
        if checks["same_preparation_binding"]
        else None,
        "checks": checks,
        "simultaneous_running_window": {
            "started_unix": overlap_start,
            "finished_unix": overlap_finish,
            "overlap_seconds": overlap_seconds,
            "configured_seconds_per_lane": EXPECTED_CONFIGURED_SECONDS,
            "required_minimum_overlap_seconds": MINIMUM_OVERLAP_SECONDS,
            "required_minimum_fraction_of_configured_window": (
                MINIMUM_OVERLAP_SECONDS / EXPECTED_CONFIGURED_SECONDS
            ),
        },
        "lanes": lanes,
    }


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize overlapping T5 fast Lane A/B engineering canaries."
    )
    parser.add_argument("--lane-a-result", type=Path, required=True)
    parser.add_argument("--lane-b-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = finalize(args.lane_a_result, args.lane_b_result)
    _write_atomic(args.output, payload)
    return 0 if payload["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
