#!/usr/bin/env python3
"""Gate 2: freeze, render, and evaluate 20 real-Isaac graph-action snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--mode", choices=("prepare", "collect", "evaluate"), required=True)
pre_args, _ = pre_parser.parse_known_args()

AppLauncher = None
if pre_args.mode == "collect":
    from isaaclab.app import AppLauncher as _AppLauncher

    AppLauncher = _AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("prepare", "collect", "evaluate"), required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--manifest")
parser.add_argument("--connectivity-root")
parser.add_argument("--scene")
parser.add_argument("--scene-usd")
parser.add_argument("--connectivity")
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

simulation_app = None
if AppLauncher is not None:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app


import hashlib
import json
import time
from typing import Any

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.evaluation import GraphEpisodeState, candidate_specs
from step3_graph_nav.io import load_yaml, read_jsonl, sha256_file, write_json, write_jsonl
from step3_graph_nav.protocol import CandidateView, GraphNavRequest
from step3_graph_nav.runner import render_candidate_views, validate_service_health


def require(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for mode={args_cli.mode}")
    return value


def prepare(config: dict[str, Any], output: Path) -> int:
    manifest = Path(require(args_cli.manifest, "--manifest")).resolve()
    connectivity_root = Path(require(args_cli.connectivity_root, "--connectivity-root")).resolve()
    selected_scenes = set(str(item) for item in config["gate2"]["scenes"])
    rows = [row for row in read_jsonl(manifest) if row.get("scene_key") in selected_scenes]
    ordinary: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    ordinary_limit = float(config["gate2"]["ordinary_abs_heading_max_deg"])
    turn_limit = float(config["gate2"]["turn_abs_heading_min_deg"])
    graph_hashes = {}
    for episode in rows:
        scene_id = str(episode["scene_key"])
        connectivity = connectivity_root / f"{scene_id}_connectivity.json"
        graph_hashes[scene_id] = sha256_file(connectivity)
        graph = MatterportGraph.load(connectivity, scene_id=scene_id)
        state = GraphEpisodeState.from_episode(graph, episode)
        path_index = 0
        while not state.goal_positive:
            path = graph.shortest_path(state.current_node, state.goal_node)
            next_node = path[1]
            specs = candidate_specs(graph, state.current_node, state.yaw_rad)
            selected = next(item for item in specs if item.target_viewpoint_id == next_node)
            absolute_heading = abs(selected.relative_heading_deg)
            category = None
            if absolute_heading <= ordinary_limit:
                category = "ordinary_move"
            elif absolute_heading >= turn_limit:
                category = "obvious_turn"
            if category:
                target = ordinary if category == "ordinary_move" else turns
                target.append(
                    {
                        "scene_id": scene_id,
                        "source_episode_id": episode["episode_id"],
                        "episode_id": state.episode_id,
                        "instruction": state.instruction,
                        "category": category,
                        "path_index": path_index,
                        "current_viewpoint_id": state.current_node,
                        "yaw_rad": state.yaw_rad,
                        "goal_positive": False,
                        "expected_action": {"action": "move", "candidate_id": selected.candidate_id},
                        "candidate_order": [item.to_mapping() for item in specs],
                    }
                )
            state.move(selected)
            path_index += 1
        specs = candidate_specs(graph, state.current_node, state.yaw_rad)
        stops.append(
            {
                "scene_id": scene_id,
                "source_episode_id": episode["episode_id"],
                "episode_id": state.episode_id,
                "instruction": state.instruction,
                "category": "stop",
                "path_index": path_index,
                "current_viewpoint_id": state.current_node,
                "yaw_rad": state.yaw_rad,
                "goal_positive": True,
                "expected_action": {"action": "stop"},
                "candidate_order": [item.to_mapping() for item in specs],
            }
        )
    counts = config["gate2"]
    plan = (
        ordinary[: int(counts["ordinary_move"])]
        + turns[: int(counts["obvious_turn"])]
        + stops[: int(counts["stop"])]
    )
    expected = int(counts["snapshots"])
    if len(plan) != expected:
        raise RuntimeError(
            f"selected train scenes produced only ordinary={len(ordinary)}, turns={len(turns)}, "
            f"stops={len(stops)}; need {counts['ordinary_move']}/{counts['obvious_turn']}/{counts['stop']}"
        )
    for index, row in enumerate(plan):
        row["snapshot_index"] = index
        row["snapshot_id"] = (
            f"offline:{index:02d}:{row['scene_id']}:{row['source_episode_id']}:"
            f"{row['path_index']}:{row['current_viewpoint_id']}"
        )
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "snapshot_plan.jsonl"
    write_jsonl(plan_path, plan)
    frozen = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_MODEL_EVALUATION",
        "source_split": "train",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "connectivity_sha256": graph_hashes,
        "snapshot_plan_sha256": sha256_file(plan_path),
        "counts": {
            "ordinary_move": sum(row["category"] == "ordinary_move" for row in plan),
            "obvious_turn": sum(row["category"] == "obvious_turn" for row in plan),
            "stop": sum(row["category"] == "stop" for row in plan),
            "total": len(plan),
        },
        "scenes": sorted({row["scene_id"] for row in plan}),
        "episode_ids": sorted({row["episode_id"] for row in plan}),
    }
    write_json(output / "snapshot_plan_manifest.json", frozen)
    print(json.dumps(frozen, separators=(",", ":")), flush=True)
    return 0


def collect(config: dict[str, Any], output: Path) -> int:
    from step3_graph_nav.isaac_scene import IsaacCandidateScene

    scene_id = require(args_cli.scene, "--scene")
    scene_usd = require(args_cli.scene_usd, "--scene-usd")
    connectivity = require(args_cli.connectivity, "--connectivity")
    graph = MatterportGraph.load(connectivity, scene_id=scene_id)
    plan = [row for row in read_jsonl(output / "snapshot_plan.jsonl") if row["scene_id"] == scene_id]
    if not plan:
        raise ValueError(f"frozen snapshot plan has no rows for scene {scene_id}")
    scene = IsaacCandidateScene(scene_usd, config, str(args_cli.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    collected = []
    for row in plan:
        snapshot_dir = output / "snapshots" / f"snapshot_{int(row['snapshot_index']):02d}"
        specs, candidates = render_candidate_views(
            scene,
            graph,
            int(row["current_viewpoint_id"]),
            float(row["yaw_rad"]),
            width=width,
            height=height,
            save_dir=snapshot_dir,
        )
        actual_order = [item.to_mapping() for item in specs]
        if actual_order != row["candidate_order"]:
            raise RuntimeError(f"candidate order drift for {row['snapshot_id']}")
        value = dict(row)
        value.update(
            {
                "scene_usd": str(Path(scene_usd).resolve()),
                "scene_usd_sha256": sha256_file(scene_usd),
                "connectivity": str(Path(connectivity).resolve()),
                "rgb_source": "isaac_render_product",
                "candidate_images": [
                    {
                        **candidate.metadata(),
                        "rgb_file": str(
                            (snapshot_dir / f"candidate_{candidate.candidate_id:02d}.jpg").relative_to(output)
                        ),
                        "rgb_sha256": hashlib.sha256(candidate.jpeg).hexdigest(),
                    }
                    for candidate in candidates
                ],
            }
        )
        collected.append(value)
    existing = []
    collected_path = output / "snapshots.jsonl"
    if collected_path.exists():
        existing = [row for row in read_jsonl(collected_path) if row["scene_id"] != scene_id]
    merged = sorted(existing + collected, key=lambda row: int(row["snapshot_index"]))
    write_jsonl(collected_path, merged)
    report = {
        "schema_version": 1,
        "mode": "collect",
        "scene_id": scene_id,
        "scene_snapshot_count": len(collected),
        "total_collected": len(merged),
        "expected": int(config["gate2"]["snapshots"]),
        "rgb_source": "isaac_render_product",
    }
    write_json(output / "collection_status.json", report)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    return 0


def evaluate(config: dict[str, Any], output: Path) -> int:
    snapshots_path = output / "snapshots.jsonl"
    snapshots = read_jsonl(snapshots_path)
    expected_count = int(config["gate2"]["snapshots"])
    if len(snapshots) != expected_count:
        raise RuntimeError(f"need exactly {expected_count} collected snapshots, found {len(snapshots)}")
    endpoint = str(config["service"]["endpoint"])
    rows = []
    with Step3GraphNavClient(endpoint, timeout_ms=int(config["service"]["timeout_ms"])) as client:
        health = client.health()
        validate_service_health(health)
        write_json(output / "service_health.json", health)
        for snapshot in snapshots:
            candidates = []
            for item in snapshot["candidate_images"]:
                candidates.append(
                    CandidateView(
                        candidate_id=int(item["candidate_id"]),
                        target_viewpoint_id=int(item["target_viewpoint_id"]),
                        relative_heading_deg=float(item["relative_heading_deg"]),
                        graph_distance_m=float(item["graph_distance_m"]),
                        jpeg=(output / item["rgb_file"]).read_bytes(),
                        width=int(item["width"]),
                        height=int(item["height"]),
                    )
                )
            request = GraphNavRequest(
                episode_id=str(snapshot["episode_id"]),
                snapshot_id=str(snapshot["snapshot_id"]),
                instruction=str(snapshot["instruction"]),
                current_viewpoint_id=int(snapshot["current_viewpoint_id"]),
                step_index=int(snapshot["path_index"]),
                candidates=tuple(candidates),
                history=(),
                timestamp=time.time(),
            )
            response = client.decide(request)
            action = response.get("action")
            rows.append(
                {
                    "snapshot_index": snapshot["snapshot_index"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "scene_id": snapshot["scene_id"],
                    "category": snapshot["category"],
                    "expected_action": snapshot["expected_action"],
                    "raw_output": response.get("raw_output", ""),
                    "parse_ok": bool(response.get("parse_ok")),
                    "candidate_id_valid": bool(response.get("candidate_id_valid")),
                    "action": action,
                    "parse_error": response.get("parse_error", ""),
                    "error_type": response.get("error_type", ""),
                    "metrics": response.get("metrics", {}),
                }
            )
    raw_path = output / "offline_raw_outputs.jsonl"
    write_jsonl(raw_path, rows)
    parse_count = sum(row["parse_ok"] for row in rows)
    invalid_ids = sum(row["error_type"] == "InvalidCandidateId" for row in rows)
    actions = [row["action"]["action"] for row in rows if row["parse_ok"]]
    expected_stop = [row["category"] == "stop" for row in rows]
    predicted_stop = [bool(row["parse_ok"] and row["action"]["action"] == "stop") for row in rows]
    latencies = sorted(float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in rows)
    ttft = sorted(float(row["metrics"].get("prefill_ttft_ms") or 0.0) for row in rows)

    def percentile(values: list[float], q: float) -> float:
        index = (len(values) - 1) * q
        low = int(index)
        high = min(low + 1, len(values) - 1)
        fraction = index - low
        return values[low] * (1.0 - fraction) + values[high] * fraction

    parse_rate = parse_count / len(rows)
    candidate_valid_rate = (len(rows) - invalid_ids) / len(rows)
    passed = bool(
        parse_rate >= float(config["gate2"]["min_json_parse_rate"])
        and candidate_valid_rate >= float(config["gate2"]["min_candidate_id_valid_rate"])
        and "move" in actions
        and "stop" in actions
    )
    summary = {
        "schema_version": 1,
        "gate": "gate2",
        "status": "PASS" if passed else "BLOCKED",
        "snapshots": len(rows),
        "json_parse_rate": parse_rate,
        "candidate_id_valid_rate": candidate_valid_rate,
        "move_outputs": actions.count("move"),
        "stop_outputs": actions.count("stop"),
        "stop_tp": sum(a and b for a, b in zip(expected_stop, predicted_stop)),
        "stop_fp": sum((not a) and b for a, b in zip(expected_stop, predicted_stop)),
        "stop_fn": sum(a and (not b) for a, b in zip(expected_stop, predicted_stop)),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "decode_tokens_per_s_mean": sum(
            float(row["metrics"].get("decode_tokens_per_s") or 0.0) for row in rows
        )
        / len(rows),
        "raw_jsonl": str(raw_path),
        "raw_jsonl_sha256": sha256_file(raw_path),
        "rgb_source": "isaac_render_product",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if passed else 2


def main() -> int:
    config = load_yaml(args_cli.config)
    output = Path(args_cli.output).resolve()
    if args_cli.mode == "prepare":
        return prepare(config, output)
    if args_cli.mode == "collect":
        return collect(config, output)
    return evaluate(config, output)


try:
    raise SystemExit(main())
finally:
    if simulation_app is not None:
        simulation_app.close()
