#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slow_benchmark.oracle_graph import MatterportGraph, habitat_pose_to_isaac
from slow_planner.base import CandidateFrontier, OrderedImage, SlowPlannerRequest
from slow_planner.client import SlowPlannerClient


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] if lower == upper else values[lower] * (upper - index) + values[upper] * (index - lower)


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure first and 30 warmed fixed-input slow-model requests.")
    parser.add_argument("--endpoint", default="tcp://10.100.100.128:8200")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--connectivity", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-model-variant", required=True)
    parser.add_argument("--expected-precision-mode", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--warm-requests", type=int, default=30)
    args = parser.parse_args()
    if len(args.image) != 4:
        raise ValueError("fixed runtime probe requires exactly four ordered RGB images")
    if args.warm_requests < 30:
        raise ValueError("formal runtime evidence requires at least 30 warmed requests")
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    episode = next(row for row in rows if str(row["benchmark_episode_id"]) == args.episode_id)
    scene_id = str(episode["scene_key"])
    graph = MatterportGraph.load(args.connectivity, scene_id=scene_id)
    node_id = graph.nearest_node(episode["start_position"])
    position, yaw = habitat_pose_to_isaac(episode["start_position"], episode["start_rotation"])
    geometries = graph.candidate_geometry(node_id, yaw)
    candidates = tuple(
        CandidateFrontier(
            frontier_id=frontier_id,
            relative_xz=(relative[0], relative[1]),
            distance_m=distance,
            bearing_deg=bearing,
        )
        for frontier_id, relative, distance, bearing, _ in geometries
    )
    image_rows = []
    ordered = []
    for index, value in enumerate(args.image):
        path = Path(value).resolve()
        with Image.open(path) as image:
            width, height = image.size
        offset = index * math.pi / 2.0
        ordered.append(
            OrderedImage(
                view_id=f"heading_{index * 90:03d}",
                pose=(position[0], position[1], position[2], yaw + offset),
                jpeg=path.read_bytes(),
                width=width,
                height=height,
            )
        )
        image_rows.append({"path": str(path), "sha256": sha256_file(path), "width": width, "height": height})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    client = SlowPlannerClient(args.endpoint, timeout_ms=180_000)
    try:
        health = client.health()
        expected = {
            "model_variant": args.expected_model_variant,
            "precision_mode": args.expected_precision_mode,
            "service_config_sha256": args.expected_config_sha256,
            "ready": True,
        }
        mismatches = {key: {"expected": value, "actual": health.get(key)} for key, value in expected.items() if health.get(key) != value}
        if mismatches:
            raise RuntimeError(f"slow service health mismatch: {mismatches}")
        result_rows = []
        for index in range(args.warm_requests + 1):
            request = SlowPlannerRequest(
                episode_id=f"fixed-runtime-probe::{args.run_id}",
                snapshot_id=f"fixed-runtime-probe::{args.run_id}::{index}",
                instruction=str(episode["instruction"]["instruction_text"]),
                ordered_images=tuple(ordered),
                candidate_frontiers=candidates,
                agent_pose=(position[0], position[1], position[2], yaw),
                visited_frontiers=(node_id,),
                compact_history=(),
            )
            started = time.perf_counter()
            decision, metrics = client.decide(request)
            client_wall_ms = (time.perf_counter() - started) * 1000.0
            row = {
                "schema_version": 1,
                "run_id": args.run_id,
                "request_index": index,
                "phase": "first_request" if index == 0 else "warmed_request",
                "decision": decision.to_mapping(),
                "metrics": metrics.to_mapping(),
                "client_wall_ms": client_wall_ms,
            }
            result_rows.append(row)
            with (output / "requests.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    finally:
        client.close()
    warm = result_rows[1:]
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "health": health,
        "fixed_input": {
            "source_split": episode["source_split"],
            "benchmark_episode_id": args.episode_id,
            "scene_id": scene_id,
            "images": image_rows,
            "candidate_frontier_count": len(candidates),
        },
        "first_request": result_rows[0],
        "warmed_request_count": len(warm),
        "warmed_client_wall_ms": distribution([float(row["client_wall_ms"]) for row in warm]),
        "warmed_ttft_ms": distribution([float(row["metrics"]["prefill_ttft_ms"]) for row in warm]),
        "warmed_end_to_end_ms": distribution([float(row["metrics"]["end_to_end_ms"]) for row in warm]),
        "warmed_peak_memory_mib": distribution([float(row["metrics"]["peak_memory_mib"]) for row in warm]),
        "structured_decision_count": sum(row["decision"]["decision"] in {"select_frontier", "target_found", "abstain"} for row in result_rows),
        "fallback_count": sum(bool(row["decision"]["fallback_used"]) for row in result_rows),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
