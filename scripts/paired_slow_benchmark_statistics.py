#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_run(path: Path, run_id: str) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("run_id") == run_id]
    result = {str(row["episode_id"]): dict(row) for row in selected}
    if len(result) != len(selected):
        raise ValueError(f"duplicate episode IDs in {path}")
    latency_path = path / "latency.jsonl"
    latency = [
        json.loads(line)
        for line in latency_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in latency:
        if row.get("run_id") == run_id:
            grouped[str(row.get("episode_id"))].append(row)
    for episode_id, row in result.items():
        slow = [item for item in grouped[episode_id] if item.get("kind") == "slow"]
        fast = [item for item in grouped[episode_id] if item.get("kind") == "fast"]
        row["_derived"] = {
            "slow_client_wall_ms_mean": statistics.fmean(float(item["client_wall_ms"]) for item in slow),
            "slow_ttft_ms_mean": statistics.fmean(float(item["metrics"]["prefill_ttft_ms"]) for item in slow),
            "slow_peak_memory_mib_max": max(float(item["metrics"]["peak_memory_mib"]) for item in slow),
            "fast_client_wall_ms_mean": statistics.fmean(float(item["client_wall_ms"]) for item in fast),
            "fast_model_latency_ms_mean": statistics.fmean(float(item["model_latency_ms"]) for item in fast),
            "fast_peak_memory_mib_max": max(float(item["peak_memory_mib"]) for item in fast),
        }
    return result


def percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def paired_bootstrap(differences: list[float], *, samples: int, seed: int) -> dict:
    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    rng = random.Random(seed)
    count = len(differences)
    estimates = []
    for _ in range(samples):
        estimates.append(statistics.fmean(differences[rng.randrange(count)] for _ in range(count)))
    estimates.sort()
    lower = estimates[round((samples - 1) * 0.025)]
    upper = estimates[round((samples - 1) * 0.975)]
    return {
        "paired_count": count,
        "mean_difference": statistics.fmean(differences),
        "ci95": [lower, upper],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def absolute_summary(values: list[float], *, samples: int, seed: int) -> dict:
    bootstrap = paired_bootstrap(values, samples=samples, seed=seed)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean_ci95": bootstrap["ci95"],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap statistics for all formal configurations.")
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_ID=RUN_DIR")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="qwen25_baseline_bf16")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    manifest_rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    order = [str(row["benchmark_episode_id"]) for row in manifest_rows]
    runs = {}
    metadata = {}
    for value in args.run:
        name, run_id, directory = value.split("=", 2)
        rows = read_run(Path(directory), run_id)
        missing = sorted(set(order) - set(rows))
        extra = sorted(set(rows) - set(order))
        if missing or extra:
            raise ValueError(f"{name} is not exactly paired: missing={len(missing)} extra={len(extra)}")
        runs[name] = rows
        metadata[name] = {"run_id": run_id, "run_dir": str(Path(directory).resolve())}
    if args.baseline not in runs:
        raise ValueError("baseline run was not supplied")

    metrics = {
        "success": lambda row: float(bool(row["success"])),
        "spl": lambda row: float(row["spl"]),
        "collisions": lambda row: float(row["collisions"]),
        "wall_seconds": lambda row: float(row["wall_seconds"]),
        "simulation_seconds": lambda row: float(row["simulation_seconds"]),
        "hold_seconds": lambda row: float(row["hold_seconds"]),
        "executed_path_m": lambda row: float(row["executed_path_m"]),
        "final_geodesic_m": lambda row: float(row["final_geodesic_m"]),
        "fast_steps": lambda row: float(row["fast_steps"]),
        "slow_decisions": lambda row: float(row["slow_decisions"]),
        "fast_effective_control_hz": lambda row: float(row["fast_effective_control_hz_excluding_slow_hold"]),
        "slow_calls_per_minute": lambda row: float(row["slow_calls_per_minute"]),
        "fast_fallback_rate": lambda row: float(row["fast_fallbacks"]) / max(1.0, float(row["fast_steps"])),
        "slow_fallback_rate": lambda row: float(row["slow_fallbacks"]) / max(1.0, float(row["slow_decisions"])),
        "target_found_tp": lambda row: float(row["target_found_tp"]),
        "target_found_fp": lambda row: float(row["target_found_fp"]),
        "target_found_fn": lambda row: float(row["target_found_fn"]),
        "frontier_geodesic_improvement_rate": lambda row: float(row["frontier_geodesic_improved"])
        / max(1.0, float(row["frontier_decisions"])),
        "repeated_frontier_rate": lambda row: float(row["repeated_frontier_decisions"])
        / max(1.0, float(row["frontier_decisions"])),
        "slow_client_wall_ms_mean": lambda row: float(row["_derived"]["slow_client_wall_ms_mean"]),
        "slow_ttft_ms_mean": lambda row: float(row["_derived"]["slow_ttft_ms_mean"]),
        "slow_peak_memory_mib_max": lambda row: float(row["_derived"]["slow_peak_memory_mib_max"]),
        "fast_client_wall_ms_mean": lambda row: float(row["_derived"]["fast_client_wall_ms_mean"]),
        "fast_model_latency_ms_mean": lambda row: float(row["_derived"]["fast_model_latency_ms_mean"]),
        "fast_peak_memory_mib_max": lambda row: float(row["_derived"]["fast_peak_memory_mib_max"]),
    }
    run_metrics = {}
    for run_index, (name, rows) in enumerate(sorted(runs.items())):
        run_metrics[name] = {
            metric: absolute_summary(
                [getter(rows[key]) for key in order],
                samples=args.bootstrap_samples,
                seed=args.seed + 10_000 + run_index * 100 + metric_index,
            )
            for metric_index, (metric, getter) in enumerate(metrics.items())
        }
    comparisons = {}
    baseline = runs[args.baseline]
    for index, (name, rows) in enumerate(sorted(runs.items())):
        if name == args.baseline:
            continue
        per_metric = {}
        for metric_index, (metric, getter) in enumerate(metrics.items()):
            differences = [getter(rows[key]) - getter(baseline[key]) for key in order]
            per_metric[metric] = paired_bootstrap(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + index * 100 + metric_index,
            )
        comparisons[f"{name}-minus-{args.baseline}"] = per_metric
    for bf16, nvfp4 in (
        ("cosmos_reason2_32b_bf16", "cosmos_reason2_32b_nvfp4"),
        ("step3_vl_10b_bf16", "step3_vl_10b_nvfp4"),
    ):
        if bf16 not in runs or nvfp4 not in runs:
            continue
        per_metric = {}
        for metric_index, (metric, getter) in enumerate(metrics.items()):
            differences = [getter(runs[nvfp4][key]) - getter(runs[bf16][key]) for key in order]
            per_metric[metric] = paired_bootstrap(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + 1000 + metric_index,
            )
        comparisons[f"{nvfp4}-minus-{bf16}"] = per_metric
    payload = {
        "schema_version": 1,
        "formal_episode_count": len(order),
        "pairing_key": "benchmark_episode_id",
        "baseline": args.baseline,
        "runs": metadata,
        "run_metrics": run_metrics,
        "comparisons": comparisons,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
