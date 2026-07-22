#!/usr/bin/env python3
"""Sample lightweight x86 CPU, memory, and one Isaac GPU into JSONL."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def _cpu_counters() -> dict[str, tuple[int, int]]:
    counters: dict[str, tuple[int, int]] = {}
    for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu"):
            continue
        if fields[0] != "cpu" and not fields[0][3:].isdigit():
            continue
        values = [int(value) for value in fields[1:]]
        idle = sum(values[index] for index in (3, 4) if index < len(values))
        counters[fields[0]] = (sum(values), idle)
    return counters


def _cpu_percent(
    previous: dict[str, tuple[int, int]], current: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    values: dict[str, float] = {}
    for name, (total, idle) in current.items():
        if name not in previous:
            continue
        prior_total, prior_idle = previous[name]
        delta_total = total - prior_total
        delta_idle = idle - prior_idle
        if delta_total > 0:
            values[name] = max(0.0, min(100.0, 100.0 * (delta_total - delta_idle) / delta_total))
    per_core = {name: value for name, value in values.items() if name != "cpu"}
    return {
        "aggregate_percent": values.get("cpu"),
        "per_core_percent": per_core,
        "busiest_core_percent": max(per_core.values()) if per_core else None,
    }


def _memory() -> dict[str, int | float | None]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        fields = raw.split()
        if fields:
            values[name] = int(fields[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return {
        "ram_total_bytes": total,
        "ram_used_bytes": max(0, total - available),
        "ram_used_percent": (100.0 * (total - available) / total) if total else None,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": max(0, swap_total - swap_free),
        "swap_used_percent": (100.0 * (swap_total - swap_free) / swap_total)
        if swap_total
        else 0.0,
    }


def _number(value: str) -> float | None:
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _gpu(index: int) -> dict[str, float | str | None]:
    fields = (
        "utilization.gpu,utilization.memory,memory.used,memory.total,"
        "temperature.gpu,power.draw,pstate"
    )
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=3.0,
    )
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi returned an unexpected row count")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != 7:
        raise RuntimeError("nvidia-smi returned an unexpected column count")
    return {
        "physical_index": index,
        "utilization_gpu_percent": _number(values[0]),
        "utilization_memory_percent": _number(values[1]),
        "memory_used_mib": _number(values[2]),
        "memory_total_mib": _number(values[3]),
        "temperature_c": _number(values[4]),
        "power_draw_w": _number(values[5]),
        "performance_state": values[6],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    arguments = parser.parse_args()
    if arguments.gpu_index < 0 or not 0.2 <= arguments.interval_sec <= 10.0:
        raise SystemExit("invalid telemetry sampler arguments")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.output.exists():
        raise SystemExit("telemetry output already exists")

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    previous = _cpu_counters()
    with arguments.output.open("x", encoding="utf-8", newline="\n") as stream:
        while not stopping:
            started = time.monotonic()
            current = _cpu_counters()
            row: dict[str, Any] = {
                "schema_version": 1,
                "sampled_unix": time.time(),
                "sampled_monotonic_sec": started,
                "cpu": _cpu_percent(previous, current),
                "memory": _memory(),
                "gpu": None,
                "errors": [],
            }
            previous = current
            try:
                row["gpu"] = _gpu(arguments.gpu_index)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                row["errors"].append(f"gpu_sample:{type(error).__name__}")
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            remaining = arguments.interval_sec - (time.monotonic() - started)
            if remaining > 0 and not stopping:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
