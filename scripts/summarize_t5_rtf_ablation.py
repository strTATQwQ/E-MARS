#!/usr/bin/env python3
"""Summarize one T5 single-Lane RTF diagnostic into machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return result
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _finite(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                result.append(value)
    return result


def _stats(values: list[Any]) -> dict[str, float | int | None]:
    finite = _finite(values)
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def _clock_window(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    valid = [
        row
        for row in rows
        if isinstance(row.get("sampled_unix"), (int, float))
        and isinstance(row.get("clock"), dict)
        and isinstance(row["clock"].get("last_clock_ns"), int)
    ]
    if len(valid) < 2:
        return {"sample_count": len(valid), "wall_duration_sec": None,
                "sim_duration_sec": None, "rtf": None}
    first, last = valid[0], valid[-1]
    wall = float(last["sampled_unix"]) - float(first["sampled_unix"])
    sim = (last["clock"]["last_clock_ns"] - first["clock"]["last_clock_ns"]) / 1e9
    return {
        "sample_count": len(valid),
        "wall_duration_sec": wall,
        "sim_duration_sec": sim,
        "rtf": sim / wall if wall > 0.0 and sim >= 0.0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dual-lane-rtf", type=float)
    arguments = parser.parse_args()
    root = arguments.result_root.resolve()
    x86 = root / "remote/x86" if (root / "remote/x86").is_dir() else root
    dgx = root / "remote/dgx" if (root / "remote/dgx").is_dir() else None
    contract = _load(x86 / "isaac_contract.json") or {}
    ablation = contract.get("rtf_ablation")
    if not isinstance(ablation, dict):
        ablation = {}

    samples = _rows(x86 / "health/engineering_canary_samples.jsonl")
    telemetry_raw = _rows(x86 / "host_telemetry.jsonl")
    canary = _load(x86 / "engineering_canary.json") or {}
    canary_window = _finite(
        [canary.get("started_unix"), canary.get("finished_unix")]
    )
    sampled_window = _finite([row.get("sampled_unix") for row in samples])
    if len(canary_window) == 2:
        sampled_start, sampled_end = min(canary_window), max(canary_window)
    elif sampled_window:
        sampled_start, sampled_end = min(sampled_window), max(sampled_window)
    else:
        sampled_start = sampled_end = None
    if sampled_start is not None and sampled_end is not None:
        telemetry = [
            row
            for row in telemetry_raw
            if _finite([row.get("sampled_unix")])
            and sampled_start <= float(row["sampled_unix"]) <= sampled_end
        ]
    else:
        telemetry = []
    per_core: dict[str, list[float]] = {}
    for row in telemetry:
        cpu = row.get("cpu")
        if not isinstance(cpu, dict) or not isinstance(cpu.get("per_core_percent"), dict):
            continue
        for name, value in cpu["per_core_percent"].items():
            finite = _finite([value])
            if finite:
                per_core.setdefault(str(name), []).append(finite[0])
    cpu_by_core = {name: _stats(values) for name, values in sorted(per_core.items())}
    busiest_core_mean = max(
        (float(value["mean"]) for value in cpu_by_core.values() if value["mean"] is not None),
        default=None,
    )
    busiest_core_maximum = max(
        (
            float(value["maximum"])
            for value in cpu_by_core.values()
            if value["maximum"] is not None
        ),
        default=None,
    )
    gpu_rows = [row.get("gpu") for row in telemetry if isinstance(row.get("gpu"), dict)]
    memory_rows = [
        row.get("memory") for row in telemetry if isinstance(row.get("memory"), dict)
    ]
    host = {
        "sample_count": len(telemetry),
        "raw_sample_count": len(telemetry_raw),
        "ready_window_start_unix": sampled_start,
        "ready_window_end_unix": sampled_end,
        "sample_error_count": sum(
            len(row.get("errors", []))
            for row in telemetry
            if isinstance(row.get("errors"), list)
        ),
        "cpu": {
            "aggregate_percent": _stats([
                row.get("cpu", {}).get("aggregate_percent")
                for row in telemetry if isinstance(row.get("cpu"), dict)
            ]),
            "busiest_core_mean_percent": busiest_core_mean,
            "busiest_core_maximum_percent": busiest_core_maximum,
            "per_core_percent": cpu_by_core,
        },
        "memory": {
            "ram_used_percent": _stats([row.get("ram_used_percent") for row in memory_rows]),
            "ram_used_bytes": _stats([row.get("ram_used_bytes") for row in memory_rows]),
            "swap_used_bytes": _stats([row.get("swap_used_bytes") for row in memory_rows]),
        },
        "gpu": {
            "utilization_percent": _stats([
                row.get("utilization_gpu_percent") for row in gpu_rows
            ]),
            "memory_used_mib": _stats([row.get("memory_used_mib") for row in gpu_rows]),
            "memory_total_mib": next(
                (row.get("memory_total_mib") for row in gpu_rows
                 if row.get("memory_total_mib") is not None), None
            ),
        },
    }

    window = _clock_window(samples)
    recovery_raw = _rows(dgx / "onboard/recovery_records.jsonl") if dgx else []
    recovery = [
        row
        for row in recovery_raw
        if sampled_start is not None
        and sampled_end is not None
        and _finite([row.get("wall_time_unix")])
        and sampled_start <= float(row["wall_time_unix"]) <= sampled_end
    ]
    spin = {
        "record_count": len(recovery),
        "raw_record_count": len(recovery_raw),
        "wall_duration_sec": _stats([row.get("spin_wall_duration_sec") for row in recovery]),
        "sim_duration_sec": _stats([row.get("spin_sim_duration_sec") for row in recovery]),
        "rtf": _stats([row.get("spin_rtf") for row in recovery]),
        "commanded_yaw_rate_rps": _stats([
            row.get("commanded_yaw_rate_rps") for row in recovery
        ]),
        "measured_yaw_rate_rps": _stats([
            row.get("measured_yaw_rate_rps") for row in recovery
        ]),
        "command_age_sec_at_start": _stats([
            row.get("command_age_sec_at_spin_start") for row in recovery
        ]),
        "command_age_sec_at_end": _stats([
            row.get("command_age_sec_at_spin_end") for row in recovery
        ]),
    }

    kit = _load(x86 / "kit_gpu_audit.json") or {}
    runtime_gpu = _load(x86 / "health/runtime_gpu_evidence.json") or {}
    fallback_patterns = (
        "falling back to cpu", "gpu dynamics disabled", "software rasterizer",
        "llvmpipe", "failed to create cuda context",
    )
    try:
        evaluator_log = (x86 / "logs/evaluator_outer.log").read_text(
            encoding="utf-8", errors="replace"
        ).lower()
    except OSError:
        evaluator_log = ""
    fallback_matches = [pattern for pattern in fallback_patterns if pattern in evaluator_log]
    acceleration = {
        "kit_rtx_gpu_identity_pass": kit.get("status") == "PASS",
        "physics_gpu_configuration_mapping_pass": runtime_gpu.get("status") == "PASS"
        and runtime_gpu.get("physics_resolved_gpu_uuid")
        == runtime_gpu.get("expected_gpu_uuid"),
        "isaac_gpu_compute_process_confirmed": runtime_gpu.get("status") == "PASS"
        and runtime_gpu.get("runtime_compute_process_uuid_observed") is True,
        "physx_runtime_gpu_execution_confirmed": runtime_gpu.get(
            "physx_runtime_gpu_execution_confirmed"
        ) is True,
        "software_fallback_log_scanned": bool(evaluator_log),
        "software_fallback_markers": fallback_matches,
        "software_fallback_absent": bool(evaluator_log) and not fallback_matches,
        "claim_scope": (
            "RTX/Isaac GPU identity, physics GPU configuration, and fallback-log "
            "scan; PhysX per-kernel execution requires an explicit runtime marker"
        ),
    }
    nested = _load(x86 / "rtf_ablation_summary.json")
    if not evaluator_log and isinstance(nested, dict) and isinstance(
        nested.get("acceleration"), dict
    ):
        acceleration = nested["acceleration"]

    rtf = window.get("rtf")
    gpu_mean = host["gpu"]["utilization_percent"]["mean"]
    gpu_memory_max = host["gpu"]["memory_used_mib"]["maximum"]
    gpu_memory_total = host["gpu"]["memory_total_mib"]
    if isinstance(rtf, (int, float)) and rtf < 0.5:
        classification = "ISAAC_HOST_THROUGHPUT_LIMITED"
    elif (
        isinstance(rtf, (int, float)) and rtf >= 0.8
        and arguments.dual_lane_rtf is not None
        and arguments.dual_lane_rtf < 0.8 * rtf
    ):
        classification = "SHARED_X86_CONTENTION"
    elif isinstance(rtf, (int, float)) and rtf >= 0.8:
        classification = "SINGLE_LANE_NEAR_REALTIME"
    else:
        classification = "INSUFFICIENT_OR_INTERMEDIATE_RTF"
    if (
        isinstance(gpu_mean, (int, float)) and gpu_mean >= 90.0
    ) or (
        isinstance(gpu_memory_max, (int, float))
        and isinstance(gpu_memory_total, (int, float))
        and gpu_memory_total > 0.0
        and gpu_memory_max / gpu_memory_total >= 0.95
    ):
        bottleneck = "GPU_RENDER_OR_RTX_SENSOR"
    elif (
        (
            isinstance(busiest_core_mean, (int, float))
            and busiest_core_mean >= 90.0
        )
        or (
            isinstance(busiest_core_maximum, (int, float))
            and busiest_core_maximum >= 98.0
        )
    ) and (not isinstance(gpu_mean, (int, float)) or gpu_mean < 80.0):
        bottleneck = "CPU_PHYSICS_OR_SENSOR_GENERATION"
    else:
        bottleneck = "MIXED_OR_UNRESOLVED"

    checks = {
        "recognized_profile": ablation.get("profile") in {
            "baseline", "lidar_off_probe", "lidar_720", "depth_stride8",
            "sensor_2p5hz",
        },
        "host_telemetry_present": len(telemetry) > 0,
        "kit_rtx_gpu_identity": acceleration["kit_rtx_gpu_identity_pass"] is True,
        "physics_gpu_configuration_mapping": acceleration[
            "physics_gpu_configuration_mapping_pass"
        ] is True,
        "isaac_gpu_compute_process": acceleration[
            "isaac_gpu_compute_process_confirmed"
        ] is True,
        "physx_runtime_gpu_execution": acceleration[
            "physx_runtime_gpu_execution_confirmed"
        ] is True,
        "no_software_fallback_marker": not acceleration["software_fallback_markers"],
    }
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "INCOMPLETE",
        "diagnostic_only": True,
        "result_root": str(root),
        "profile": ablation,
        "engineering_window": window,
        "spin": spin,
        "host": host,
        "acceleration": acceleration,
        "classification": classification,
        "bottleneck": bottleneck,
        "dual_lane_reference_rtf": arguments.dual_lane_rtf,
        "checks": checks,
        "recorded_unix": time.time(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
