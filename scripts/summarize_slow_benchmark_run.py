#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], probability: float) -> float | None:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def distribution(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize one slow-model benchmark JSONL run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(args.run_dir).resolve()
    episodes = [row for row in read_jsonl(root / "episodes.jsonl") if row.get("run_id") == args.run_id]
    latency = [row for row in read_jsonl(root / "latency.jsonl") if row.get("run_id") == args.run_id]
    slow = [row for row in latency if row.get("kind") == "slow"]
    fast = [row for row in latency if row.get("kind") == "fast" and "client_wall_ms" in row]
    stable_decode_rate_rows = [
        row
        for row in slow
        if int(row.get("metrics", {}).get("output_token_count", 0)) >= 2
        and float(row.get("metrics", {}).get("decode_ms", 0.0)) >= 1.0
    ]
    tp = sum(int(row.get("target_found_tp", 0)) for row in episodes)
    fp = sum(int(row.get("target_found_fp", 0)) for row in episodes)
    fn = sum(int(row.get("target_found_fn", 0)) for row in episodes)
    selected_slow = [row for row in slow if row.get("decision", {}).get("decision") == "select_frontier"]
    frontier_valid = [bool(row.get("evaluation", {}).get("frontier_valid")) for row in selected_slow]
    frontier_improved = [
        float(row.get("evaluation", {}).get("goal_geodesic_delta_m")) > 0.0
        for row in selected_slow
        if row.get("evaluation", {}).get("goal_geodesic_delta_m") is not None
    ]
    repeated_frontiers = [bool(row.get("evaluation", {}).get("repeated_frontier")) for row in selected_slow]
    videos = [row.get("video") for row in episodes if isinstance(row.get("video"), dict)]
    collision_episodes = sum(int(row.get("collisions", 0)) > 0 for row in episodes)
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "episode_count": len(episodes),
        "scene_count": len({row.get("scene_id") for row in episodes}),
        "success_rate": sum(bool(row.get("success")) for row in episodes) / len(episodes) if episodes else None,
        "mean_spl": statistics.fmean(float(row.get("spl", 0.0)) for row in episodes) if episodes else None,
        "episode_wall_seconds": distribution([row.get("wall_seconds", 0.0) for row in episodes]),
        "executed_path_m": distribution([row.get("executed_path_m", 0.0) for row in episodes]),
        "slow_decisions": distribution([row.get("slow_decisions", 0) for row in episodes]),
        "fast_steps": distribution([row.get("fast_steps", 0) for row in episodes]),
        "simulation_seconds": distribution([row.get("simulation_seconds", 0.0) for row in episodes]),
        "hold_seconds": distribution([row.get("hold_seconds", 0.0) for row in episodes]),
        "hold_fraction_wall": distribution([row.get("hold_fraction_wall", 0.0) for row in episodes]),
        "slow_calls_per_minute": distribution([row.get("slow_calls_per_minute", 0.0) for row in episodes]),
        "fast_effective_control_hz_excluding_slow_hold": distribution(
            [row.get("fast_effective_control_hz_excluding_slow_hold", 0.0) for row in episodes]
        ),
        "fast_control_hz_during_slow_hold": 0.0,
        "fast_policy_request_hz_outside_slow_hold": distribution(
            [1000.0 / float(row["client_wall_ms"]) for row in fast if float(row.get("client_wall_ms", 0.0)) > 0.0]
        ),
        "collisions": sum(int(row.get("collisions", 0)) for row in episodes),
        "collision_episode_rate": collision_episodes / len(episodes) if episodes else None,
        "failure_reasons": dict(sorted(Counter(str(row.get("failure_reason") or "success") for row in episodes).items())),
        "target_found": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "slow_client_wall_ms": distribution([row.get("client_wall_ms", 0.0) for row in slow]),
        "slow_preprocessing_ms": distribution([row.get("metrics", {}).get("preprocessing_ms", 0.0) for row in slow]),
        "slow_visual_token_count": distribution([row.get("metrics", {}).get("visual_token_count", 0) for row in slow]),
        "slow_input_token_count": distribution([row.get("metrics", {}).get("input_token_count", 0) for row in slow]),
        "slow_output_token_count": distribution([row.get("metrics", {}).get("output_token_count", 0) for row in slow]),
        "slow_image_count": distribution([row.get("metrics", {}).get("image_count", 0) for row in slow]),
        "slow_ttft_ms": distribution([row.get("metrics", {}).get("prefill_ttft_ms", 0.0) for row in slow]),
        "slow_decode_tokens_per_s": distribution(
            [row.get("metrics", {}).get("decode_tokens_per_s", 0.0) for row in stable_decode_rate_rows]
        ),
        "slow_peak_memory_mib": distribution([row.get("metrics", {}).get("peak_memory_mib", 0.0) for row in slow]),
        "fast_client_wall_ms": distribution([row.get("client_wall_ms", 0.0) for row in fast]),
        "fast_model_latency_ms": distribution([row.get("model_latency_ms", 0.0) for row in fast]),
        "fast_fallback_rate": sum(bool(row.get("fallback")) for row in fast) / len(fast) if fast else None,
        "fast_waypoint_clamp_rate": sum(bool(row.get("waypoint_clamped")) for row in fast) / len(fast) if fast else None,
        "slow_model_navigation": {
            "selected_frontiers": len(selected_slow),
            "valid_frontier_rate": sum(frontier_valid) / len(frontier_valid) if frontier_valid else None,
            "geodesic_improvement_rate": sum(frontier_improved) / len(frontier_improved) if frontier_improved else None,
            "repeated_frontier_rate": sum(repeated_frontiers) / len(repeated_frontiers) if repeated_frontiers else None,
            "geodesic_delta_m": distribution(
                [
                    row.get("evaluation", {}).get("goal_geodesic_delta_m")
                    for row in selected_slow
                    if row.get("evaluation", {}).get("goal_geodesic_delta_m") is not None
                ]
            ),
            "fallback_rate": sum(bool(row.get("decision", {}).get("fallback_used")) for row in slow) / len(slow)
            if slow
            else None,
        },
        "video_evidence": {
            "episode_video_count": len(videos),
            "bytes": sum(int(video.get("bytes", 0)) for video in videos),
            "frame_count": sum(int(video.get("frame_count", 0)) for video in videos),
            "encoding_seconds_excluded_from_episode_wall": distribution(
                [video.get("encoding_seconds_excluded_from_episode_wall", 0.0) for video in videos]
            ),
        },
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
