#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


NUMERIC_FIELDS = (
    "gpu_util_pct",
    "gpu_mem_util_pct",
    "power_w",
    "temperature_c",
    "temp_c",
    "sm_clock_mhz",
    "load1",
    "ram_used_mib",
    "ram_total_mib",
    "llama_rss_kib",
    "mem_available_kib",
    "step_rss_kib",
    "grounded_sam_rss_kib",
    "omninav_rss_kib",
    "internnav_rss_kib",
)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(quantile * len(ordered))))
    return round(ordered[rank - 1], 3)


def summarize(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    active_rows = []
    for row in rows:
        try:
            if float(row.get("gpu_util_pct", 0.0)) > 0.0:
                active_rows.append(row)
        except (TypeError, ValueError):
            continue
    result = {"schema_version": 1, "samples": len(rows), "active_samples": len(active_rows), "metrics": {}}
    for field in NUMERIC_FIELDS:
        values = []
        active_values = []
        for row in rows:
            try:
                values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
        for row in active_rows:
            try:
                active_values.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                pass
        if values:
            result["metrics"][field] = {
                "mean": round(sum(values) / len(values), 3),
                "p95": percentile(values, 0.95),
                "max": round(max(values), 3),
                "active_mean": round(sum(active_values) / len(active_values), 3) if active_values else None,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge and summarize DGX performance CSV files.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize([Path(value) for value in args.inputs])
    (output / "dgx_performance_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# DGX Performance Summary", "", f"- samples: {summary['samples']}", f"- active samples: {summary['active_samples']}"]
    for field, values in summary["metrics"].items():
        lines.append(
            f"- {field}: mean={values['mean']}, active_mean={values['active_mean']}, p95={values['p95']}, max={values['max']}"
        )
    (output / "dgx_performance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
