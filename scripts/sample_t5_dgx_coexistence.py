#!/usr/bin/env python3
"""Boundedly sample InternVLA + Step3 coexistence memory without control authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


_NVIDIA_NOT_APPLICABLE = frozenset(
    {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}
)


def _read_kib_fields(path: Path, wanted: Sequence[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, remainder = line.partition(":")
        if separator and key in wanted:
            token = remainder.strip().split()[0]
            values[key] = int(token)
    return values


def _process_sample(pid: int) -> dict[str, Any]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.is_file():
        return {"pid": pid, "alive": False}
    values = _read_kib_fields(
        status_path, ("VmRSS", "VmHWM", "VmSwap", "RssAnon", "RssFile")
    )
    return {
        "pid": pid,
        "alive": True,
        "rss_kib": values.get("VmRSS"),
        "rss_high_water_kib": values.get("VmHWM"),
        "swap_kib": values.get("VmSwap", 0),
        "rss_anon_kib": values.get("RssAnon"),
        "rss_file_kib": values.get("RssFile"),
    }


def _system_memory_sample() -> dict[str, Any]:
    values = _read_kib_fields(
        Path("/proc/meminfo"),
        ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"),
    )
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    oom_kill_count = None
    for line in Path("/proc/vmstat").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(" ")
        if separator and key == "oom_kill":
            oom_kill_count = int(value.strip())
            break
    return {
        "mem_total_kib": values.get("MemTotal"),
        "mem_available_kib": values.get("MemAvailable"),
        "swap_total_kib": swap_total,
        "swap_free_kib": swap_free,
        "swap_used_kib": max(0, swap_total - swap_free),
        "oom_kill_count": oom_kill_count,
    }


def _float_or_none(value: str) -> float | None:
    normalized = value.strip()
    if normalized == "" or normalized in _NVIDIA_NOT_APPLICABLE:
        return None
    try:
        number = float(normalized)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _gpu_memory_accounting_status(
    used_raw: str,
    total_raw: str,
    *,
    used_mib: float | None,
    total_mib: float | None,
) -> str:
    if used_mib is not None and total_mib is not None:
        return "observed"
    if (
        used_raw.strip() in _NVIDIA_NOT_APPLICABLE
        and total_raw.strip() in _NVIDIA_NOT_APPLICABLE
    ):
        # GB10 exposes unified system memory and reports dedicated framebuffer
        # accounting as N/A.  Preserve the missing values instead of inventing
        # a VRAM total from system RAM.
        return "not_applicable_unified"
    return "unavailable"


def _gpu_sample(gpu_index: int) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = [row for row in query.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi returned an unexpected GPU row count")
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError("nvidia-smi returned an unexpected GPU column count")
    memory_used_mib = _float_or_none(fields[3])
    memory_total_mib = _float_or_none(fields[4])
    processes: list[dict[str, Any]] = []
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if process_query.returncode == 0:
        for row in process_query.stdout.splitlines():
            if not row.strip():
                continue
            parts = [value.strip() for value in row.split(",", 2)]
            if len(parts) != 3:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            processes.append(
                {
                    "pid": pid,
                    "process_name": Path(parts[1]).name,
                    "used_gpu_memory_mib": _float_or_none(parts[2]),
                }
            )
    return {
        "gpu_index": int(fields[0]),
        "gpu_uuid": fields[1],
        "utilization_percent": _float_or_none(fields[2]),
        "memory_used_mib": memory_used_mib,
        "memory_total_mib": memory_total_mib,
        "memory_accounting_status": _gpu_memory_accounting_status(
            fields[3],
            fields[4],
            used_mib=memory_used_mib,
            total_mib=memory_total_mib,
        ),
        "compute_processes": processes,
    }


def sample_once(
    *, internvla_pid: int, step3_pid: int, gpu_index: int
) -> dict[str, Any]:
    gpu: dict[str, Any]
    try:
        gpu = _gpu_sample(gpu_index)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        gpu = {"gpu_index": gpu_index, "sample_error": type(exc).__name__}
    return {
        "schema_version": 1,
        "monotonic_s": time.monotonic(),
        "wall_time_s": time.time(),
        "system_memory": _system_memory_sample(),
        "processes": {
            "internvla": _process_sample(internvla_pid),
            "step3": _process_sample(step3_pid),
        },
        "gpu": gpu,
    }


def _numbers(samples: Sequence[Mapping[str, Any]], path: Sequence[str]) -> list[float]:
    values = []
    for sample in samples:
        current: Any = sample
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            number = float(current)
            if math.isfinite(number):
                values.append(number)
    return values


def _gpu_memory_status(sample: Mapping[str, Any]) -> str:
    gpu = sample.get("gpu")
    if not isinstance(gpu, Mapping):
        return "unavailable"
    status = str(gpu.get("memory_accounting_status") or "")
    if status in {"observed", "not_applicable_unified"}:
        return status
    used = gpu.get("memory_used_mib")
    total = gpu.get("memory_total_mib")
    if all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in (used, total)
    ):
        return "observed"
    return "unavailable"


def summarize(
    samples: Sequence[Mapping[str, Any]],
    *,
    internvla_pid: int,
    step3_pid: int,
    gpu_index: int,
    minimum_samples: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("coexistence sampling produced no samples")
    process_alive = {
        name: all(
            bool(sample.get("processes", {}).get(name, {}).get("alive"))
            for sample in samples
        )
        for name in ("internvla", "step3")
    }
    process_swap_peak = {
        name: max(
            _numbers(samples, ("processes", name, "swap_kib")) or [float("inf")]
        )
        for name in ("internvla", "step3")
    }
    process_rss_peak = {
        name: max(_numbers(samples, ("processes", name, "rss_kib")) or [0.0])
        for name in ("internvla", "step3")
    }
    system_swap = _numbers(samples, ("system_memory", "swap_used_kib"))
    available = _numbers(samples, ("system_memory", "mem_available_kib"))
    oom_kills = _numbers(samples, ("system_memory", "oom_kill_count"))
    gpu_used = _numbers(samples, ("gpu", "memory_used_mib"))
    gpu_total = _numbers(samples, ("gpu", "memory_total_mib"))
    gpu_utilization = _numbers(samples, ("gpu", "utilization_percent"))
    gpu_rows = [sample.get("gpu") for sample in samples]
    gpu_uuids = [
        str(gpu.get("gpu_uuid") or "")
        for gpu in gpu_rows
        if isinstance(gpu, Mapping)
    ]
    gpu_indices = [
        gpu.get("gpu_index") for gpu in gpu_rows if isinstance(gpu, Mapping)
    ]
    gpu_memory_statuses = [_gpu_memory_status(sample) for sample in samples]
    if gpu_memory_statuses and all(
        status == "observed" for status in gpu_memory_statuses
    ):
        gpu_memory_accounting_status = "observed"
    elif gpu_memory_statuses and all(
        status == "not_applicable_unified" for status in gpu_memory_statuses
    ):
        gpu_memory_accounting_status = "not_applicable_unified"
    else:
        gpu_memory_accounting_status = "unavailable"
    gpu_memory_accounting_valid = (
        gpu_memory_accounting_status == "observed"
        and len(gpu_used) == len(samples)
        and len(gpu_total) == len(samples)
        and all(
            0.0 <= used <= total and total > 0.0
            for used, total in zip(gpu_used, gpu_total)
        )
    ) or (
        gpu_memory_accounting_status == "not_applicable_unified"
        and not gpu_used
        and not gpu_total
    )
    process_rss_observed = {
        name: len(_numbers(samples, ("processes", name, "rss_kib"))) == len(samples)
        for name in ("internvla", "step3")
    }
    checks = {
        "minimum_samples": len(samples) >= minimum_samples,
        "internvla_alive_throughout": process_alive["internvla"],
        "step3_alive_throughout": process_alive["step3"],
        "internvla_rss_observed": process_rss_observed["internvla"],
        "step3_rss_observed": process_rss_observed["step3"],
        "internvla_process_swap_zero": process_swap_peak["internvla"] == 0.0,
        "step3_process_swap_zero": process_swap_peak["step3"] == 0.0,
        "system_memory_observed": len(available) == len(samples),
        "system_swap_observed": len(system_swap) == len(samples),
        "system_swap_did_not_increase": len(system_swap) == len(samples)
        and max(system_swap) <= system_swap[0],
        "oom_kill_counter_observed": len(oom_kills) == len(samples),
        "system_oom_kill_did_not_increase": len(oom_kills) == len(samples)
        and max(oom_kills) <= oom_kills[0],
        "gpu_uuid_observed_and_stable": len(gpu_uuids) == len(samples)
        and all(gpu_uuids)
        and all(uuid not in _NVIDIA_NOT_APPLICABLE for uuid in gpu_uuids)
        and len(set(gpu_uuids)) == 1,
        "gpu_index_observed_and_stable": len(gpu_indices) == len(samples)
        and all(index == gpu_index for index in gpu_indices),
        "gpu_utilization_observed": len(gpu_utilization) == len(samples)
        and all(0.0 <= value <= 100.0 for value in gpu_utilization),
        "gpu_memory_accounting_valid": gpu_memory_accounting_valid,
    }
    return {
        "schema_version": 1,
        "kind": "t5_dgx_internvla_step3_coexistence_memory",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_count": len(samples),
        "gpu_index": gpu_index,
        "pids": {"internvla": internvla_pid, "step3": step3_pid},
        "checks": checks,
        "process_rss_peak_kib": process_rss_peak,
        "process_swap_peak_kib": process_swap_peak,
        "system_mem_available_min_kib": min(available) if available else None,
        "system_swap_used_initial_kib": system_swap[0] if system_swap else None,
        "system_swap_used_peak_kib": max(system_swap) if system_swap else None,
        "system_oom_kill_initial": oom_kills[0] if oom_kills else None,
        "system_oom_kill_peak": max(oom_kills) if oom_kills else None,
        "gpu_uuid": gpu_uuids[0] if gpu_uuids else None,
        "gpu_utilization_peak_percent": max(gpu_utilization)
        if gpu_utilization
        else None,
        "gpu_memory_accounting_status": gpu_memory_accounting_status,
        "gpu_memory_used_peak_mib": max(gpu_used)
        if gpu_memory_accounting_status == "observed" and gpu_used
        else None,
        "gpu_memory_total_mib": gpu_total[0]
        if gpu_memory_accounting_status == "observed" and gpu_total
        else None,
        "memory_evidence_basis": (
            "system_and_process_rss_swap_oom_with_gpu_identity_and_utilization"
        ),
        "oom_evidence_scope": "owned_process_survival_plus_system_oom_kill_counter",
        "motion_authority": "none",
        "recorded_wall_time_s": time.time(),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize_preload(
    samples: Sequence[Mapping[str, Any]], *, stop_observed: bool
) -> dict[str, Any]:
    system_swap = _numbers(samples, ("system_memory", "swap_used_kib"))
    available = _numbers(samples, ("system_memory", "mem_available_kib"))
    oom_kills = _numbers(samples, ("system_memory", "oom_kill_count"))
    checks = {
        "minimum_samples": len(samples) >= 2,
        "explicit_stop_observed": stop_observed,
        "system_memory_observed": len(available) == len(samples),
        "system_swap_observed": len(system_swap) == len(samples),
        "system_swap_did_not_increase": len(system_swap) == len(samples)
        and max(system_swap) <= system_swap[0],
        "oom_kill_counter_observed": len(oom_kills) == len(samples),
        "system_oom_kill_did_not_increase": len(oom_kills) == len(samples)
        and max(oom_kills) <= oom_kills[0],
    }
    return {
        "schema_version": 1,
        "kind": "t5_dgx_model_preload_system_memory",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sample_count": len(samples),
        "checks": checks,
        "system_mem_available_min_kib": min(available) if available else None,
        "system_swap_used_initial_kib": system_swap[0] if system_swap else None,
        "system_swap_used_peak_kib": max(system_swap) if system_swap else None,
        "system_oom_kill_initial": oom_kills[0] if oom_kills else None,
        "system_oom_kill_peak": max(oom_kills) if oom_kills else None,
        "recorded_wall_time_s": time.time(),
    }


def run_preload_monitor(
    *,
    samples_path: Path,
    summary_path: Path,
    stop_file: Path,
    interval_sec: float,
    max_duration_sec: float,
) -> int:
    if samples_path.exists() or summary_path.exists() or stop_file.exists():
        raise ValueError("preload monitor paths must be fresh")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + max_duration_sec
    next_sample = started
    stop_observed = False
    with samples_path.open("x", encoding="utf-8") as handle:
        while True:
            sample = {
                "schema_version": 1,
                "monotonic_s": time.monotonic(),
                "wall_time_s": time.time(),
                "system_memory": _system_memory_sample(),
            }
            samples.append(sample)
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            handle.flush()
            if stop_file.exists() or stop_file.is_symlink():
                if stop_file.is_symlink() or not stop_file.is_file():
                    raise ValueError("preload stop marker must be a regular file")
                stop_observed = True
                break
            now = time.monotonic()
            if now >= deadline:
                break
            next_sample += interval_sec
            time.sleep(min(max(0.0, next_sample - now), deadline - now))
        os.fsync(handle.fileno())
    summary = summarize_preload(samples, stop_observed=stop_observed)
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 75


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internvla-pid", type=int)
    parser.add_argument("--step3-pid", type=int)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preload-stop-file", type=Path)
    args = parser.parse_args()
    if not 0.2 <= args.interval_sec <= 10.0:
        raise ValueError("interval-sec must be in [0.2, 10]")
    if args.preload_stop_file is not None:
        if args.internvla_pid is not None or args.step3_pid is not None:
            raise ValueError("preload monitor does not accept model PIDs")
        if not 5.0 <= args.duration_sec <= 3600.0:
            raise ValueError("preload monitor duration-sec must be in [5, 3600]")
        return run_preload_monitor(
            samples_path=args.samples,
            summary_path=args.summary,
            stop_file=args.preload_stop_file,
            interval_sec=args.interval_sec,
            max_duration_sec=args.duration_sec,
        )
    if args.internvla_pid is None or args.step3_pid is None:
        raise ValueError("both model PIDs are required outside preload mode")
    if args.internvla_pid <= 1 or args.step3_pid <= 1:
        raise ValueError("both owned process PIDs must be greater than one")
    if args.internvla_pid == args.step3_pid:
        raise ValueError("InternVLA and Step3 must be distinct processes")
    if args.gpu_index < 0:
        raise ValueError("gpu-index must be non-negative")
    if not 5.0 <= args.duration_sec <= 900.0:
        raise ValueError("duration-sec must be in [5, 900]")
    if args.samples.exists() or args.summary.exists():
        raise ValueError("coexistence output paths must not already exist")
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    started = time.monotonic()
    deadline = started + args.duration_sec
    next_sample = started
    with args.samples.open("x", encoding="utf-8") as handle:
        while True:
            value = sample_once(
                internvla_pid=args.internvla_pid,
                step3_pid=args.step3_pid,
                gpu_index=args.gpu_index,
            )
            samples.append(value)
            handle.write(json.dumps(value, separators=(",", ":")) + "\n")
            handle.flush()
            now = time.monotonic()
            if now >= deadline:
                break
            # Keep an absolute cadence.  Sleeping a full interval *after* two
            # nvidia-smi calls would make a valid 60 s / 1 Hz run produce fewer
            # than 60 samples solely because of sampler overhead.
            next_sample += args.interval_sec
            time.sleep(min(max(0.0, next_sample - now), deadline - now))
        os.fsync(handle.fileno())
    minimum_samples = max(2, int(args.duration_sec / args.interval_sec))
    summary = summarize(
        samples,
        internvla_pid=args.internvla_pid,
        step3_pid=args.step3_pid,
        gpu_index=args.gpu_index,
        minimum_samples=minimum_samples,
    )
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 75


if __name__ == "__main__":
    raise SystemExit(main())
