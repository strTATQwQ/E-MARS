#!/usr/bin/env python3
"""Gate 2A: freeze, collect, and evaluate move-only NavigationPolicy snapshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--mode", choices=("prepare", "collect", "evaluate"), required=True)
pre_args, _ = pre.parse_known_args()

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
parser.add_argument("--reuse-root")
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = None
if AppLauncher is not None:
    launcher = AppLauncher(args)
    simulation_app = launcher.app

import hashlib
import json
import shutil
import time
from collections import defaultdict
from typing import Any

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.evaluation import GraphEpisodeState, candidate_specs
from step3_graph_nav.io import load_yaml, read_jsonl, sha256_file, write_json, write_jsonl
from step3_graph_nav.protocol import CandidateView, GraphNavRequest
from step3_graph_nav.runner import render_candidate_views, validate_service_health


def require(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for mode={args.mode}")
    return value


def action_summary(relative_heading_deg: float, distance_m: float) -> str:
    if relative_heading_deg < -25.0:
        turn = "turned right"
    elif relative_heading_deg > 25.0:
        turn = "turned left"
    else:
        turn = "continued forward"
    return f"{turn} by {abs(relative_heading_deg):.1f} degrees and moved {distance_m:.2f} m"


def prepare(config: dict[str, Any], output: Path) -> int:
    manifest = Path(require(args.manifest, "--manifest")).resolve()
    connectivity_root = Path(require(args.connectivity_root, "--connectivity-root")).resolve()
    allowed_scenes = set(str(item) for item in config["gate2a"]["scenes"])
    episodes = [row for row in read_jsonl(manifest) if str(row.get("scene_key")) in allowed_scenes]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    graph_hashes = {}
    for episode in episodes:
        scene_id = str(episode["scene_key"])
        graph_path = connectivity_root / f"{scene_id}_connectivity.json"
        graph_hashes[scene_id] = sha256_file(graph_path)
        graph = MatterportGraph.load(graph_path, scene_id=scene_id)
        state = GraphEpisodeState.from_episode(graph, episode)
        history: list[str] = []
        path_index = 0
        while not state.goal_positive:
            specs = candidate_specs(graph, state.current_node, state.yaw_rad)
            shortest_path = graph.shortest_path(state.current_node, state.goal_node)
            expected_target = shortest_path[1]
            expected = next(item for item in specs if item.target_viewpoint_id == expected_target)
            by_episode[state.episode_id].append(
                {
                    "scene_id": scene_id,
                    "source_episode_id": int(episode["episode_id"]),
                    "episode_id": state.episode_id,
                    "instruction": state.instruction,
                    "path_index": path_index,
                    "current_viewpoint_id": state.current_node,
                    "yaw_rad": state.yaw_rad,
                    "current_ne_m": state.ne_m,
                    "expected_candidate_id": expected.candidate_id,
                    "candidate_order": [item.to_mapping() for item in specs],
                    "candidate_ne_m": {
                        str(item.candidate_id): graph.shortest_distance(
                            item.target_viewpoint_id, state.goal_node
                        )
                        for item in specs
                    },
                    "action_history": list(history[-4:]),
                }
            )
            state.move(expected)
            history.append(action_summary(expected.relative_heading_deg, expected.graph_distance_m))
            path_index += 1

    target_count = int(config["gate2a"]["snapshots"])
    plan: list[dict[str, Any]] = []
    episode_keys = sorted(by_episode)
    round_index = 0
    while len(plan) < target_count:
        added = False
        for episode_id in episode_keys:
            rows = by_episode[episode_id]
            if round_index < len(rows):
                plan.append(rows[round_index])
                added = True
                if len(plan) == target_count:
                    break
        if not added:
            break
        round_index += 1
    if len(plan) != target_count:
        raise RuntimeError(f"only {len(plan)} non-terminal train states; need {target_count}")
    for index, row in enumerate(plan):
        row["snapshot_index"] = index
        row["snapshot_id"] = (
            f"nav2a:{index:02d}:{row['scene_id']}:{row['source_episode_id']}:"
            f"{row['path_index']}:{row['current_viewpoint_id']}"
        )
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "snapshot_plan.jsonl"
    write_jsonl(plan_path, plan)
    manifest_value = {
        "schema_version": 2,
        "gate": "gate2a",
        "status": "FROZEN_BEFORE_MODEL_EVALUATION",
        "source_split": "train",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "connectivity_sha256": graph_hashes,
        "snapshot_plan_sha256": sha256_file(plan_path),
        "snapshot_count": len(plan),
        "all_non_terminal": True,
        "scenes": sorted({row["scene_id"] for row in plan}),
        "episode_ids": sorted({row["episode_id"] for row in plan}),
    }
    write_json(output / "snapshot_plan_manifest.json", manifest_value)
    print(json.dumps(manifest_value, separators=(",", ":")), flush=True)
    return 0


def _reuse_index(root: Path | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    if root is None or not (root / "snapshots.jsonl").exists():
        return {}
    result = {}
    for row in read_jsonl(root / "snapshots.jsonl"):
        key = (
            row.get("scene_id"),
            str(row.get("episode_id")),
            int(row.get("path_index")),
            int(row.get("current_viewpoint_id")),
        )
        result[key] = row
    return result


def collect(config: dict[str, Any], output: Path) -> int:
    from step3_graph_nav.isaac_scene import IsaacCandidateScene

    scene_id = require(args.scene, "--scene")
    scene_usd = require(args.scene_usd, "--scene-usd")
    connectivity = require(args.connectivity, "--connectivity")
    graph = MatterportGraph.load(connectivity, scene_id=scene_id)
    plan = [row for row in read_jsonl(output / "snapshot_plan.jsonl") if row["scene_id"] == scene_id]
    reuse_root = Path(args.reuse_root).resolve() if args.reuse_root else None
    reuse = _reuse_index(reuse_root)
    scene = IsaacCandidateScene(scene_usd, config, str(args.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    collected = []
    reused_count = 0
    rendered_count = 0
    for row in plan:
        snapshot_dir = output / "snapshots" / f"snapshot_{int(row['snapshot_index']):02d}"
        key = (
            row["scene_id"],
            str(row["episode_id"]),
            int(row["path_index"]),
            int(row["current_viewpoint_id"]),
        )
        old = reuse.get(key)
        candidate_images = []
        if old and old.get("candidate_order") == row["candidate_order"]:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for item in old["candidate_images"]:
                source = reuse_root / item["rgb_file"]
                target = snapshot_dir / f"candidate_{int(item['candidate_id']):02d}.jpg"
                shutil.copy2(source, target)
                candidate_images.append(
                    {
                        **{key: item[key] for key in (
                            "candidate_id", "target_viewpoint_id", "relative_heading_deg",
                            "graph_distance_m", "width", "height"
                        )},
                        "rgb_file": str(target.relative_to(output)),
                        "rgb_sha256": sha256_file(target),
                        "reused_from": str(source),
                    }
                )
            reused_count += 1
        else:
            specs, candidates = render_candidate_views(
                scene,
                graph,
                int(row["current_viewpoint_id"]),
                float(row["yaw_rad"]),
                width=width,
                height=height,
                save_dir=snapshot_dir,
            )
            if [item.to_mapping() for item in specs] != row["candidate_order"]:
                raise RuntimeError(f"candidate order drift for {row['snapshot_id']}")
            candidate_images = [
                {
                    **candidate.metadata(),
                    "rgb_file": str(
                        (snapshot_dir / f"candidate_{candidate.candidate_id:02d}.jpg").relative_to(output)
                    ),
                    "rgb_sha256": hashlib.sha256(candidate.jpeg).hexdigest(),
                    "reused_from": "",
                }
                for candidate in candidates
            ]
            rendered_count += 1
        value = dict(row)
        value.update(
            {
                "scene_usd": str(Path(scene_usd).resolve()),
                "scene_usd_sha256": sha256_file(scene_usd),
                "connectivity": str(Path(connectivity).resolve()),
                "rgb_source": "isaac_render_product",
                "candidate_images": candidate_images,
            }
        )
        collected.append(value)
    existing_path = output / "snapshots.jsonl"
    existing = []
    if existing_path.exists():
        existing = [row for row in read_jsonl(existing_path) if row["scene_id"] != scene_id]
    merged = sorted(existing + collected, key=lambda row: int(row["snapshot_index"]))
    write_jsonl(existing_path, merged)
    report = {
        "schema_version": 2,
        "gate": "gate2a",
        "scene_id": scene_id,
        "scene_snapshot_count": len(collected),
        "total_collected": len(merged),
        "reused_snapshot_count": reused_count,
        "newly_rendered_snapshot_count": rendered_count,
        "rgb_source": "isaac_render_product",
    }
    write_json(output / f"collection_{scene_id}.json", report)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    return 0


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    fraction = index - low
    return values[low] * (1.0 - fraction) + values[high] * fraction


def evaluate(config: dict[str, Any], output: Path) -> int:
    snapshots = read_jsonl(output / "snapshots.jsonl")
    expected = int(config["gate2a"]["snapshots"])
    if len(snapshots) != expected:
        raise RuntimeError(f"Gate 2A requires {expected} snapshots, found {len(snapshots)}")
    rows = []
    with Step3GraphNavClient(
        str(config["service"]["endpoint"]), timeout_ms=int(config["service"]["timeout_ms"])
    ) as client:
        health = client.health()
        validate_service_health(health)
        write_json(output / "service_health.json", health)
        for snapshot in snapshots:
            candidates = tuple(
                CandidateView(
                    candidate_id=int(item["candidate_id"]),
                    target_viewpoint_id=int(item["target_viewpoint_id"]),
                    relative_heading_deg=float(item["relative_heading_deg"]),
                    graph_distance_m=float(item["graph_distance_m"]),
                    jpeg=(output / item["rgb_file"]).read_bytes(),
                    width=int(item["width"]),
                    height=int(item["height"]),
                )
                for item in snapshot["candidate_images"]
            )
            request = GraphNavRequest(
                episode_id=str(snapshot["episode_id"]),
                snapshot_id=str(snapshot["snapshot_id"]),
                instruction=str(snapshot["instruction"]),
                current_viewpoint_id=int(snapshot["current_viewpoint_id"]),
                step_index=int(snapshot["path_index"]),
                candidates=candidates,
                history=tuple(str(item) for item in snapshot["action_history"]),
            )
            response = client.navigate(request)
            action = response.get("action")
            selected = int(action["candidate_id"]) if response.get("parse_ok") else None
            ranked = [int(item) for item in response.get("metrics", {}).get("ranked_candidate_ids", [])]
            rows.append(
                {
                    "snapshot_index": snapshot["snapshot_index"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "episode_id": snapshot["episode_id"],
                    "path_index": snapshot["path_index"],
                    "expected_candidate_id": snapshot["expected_candidate_id"],
                    "candidate_count": len(candidates),
                    "current_ne_m": snapshot["current_ne_m"],
                    "candidate_ne_m": snapshot["candidate_ne_m"],
                    "raw_output": response.get("raw_output", ""),
                    "parse_ok": bool(response.get("parse_ok")),
                    "candidate_id_valid": bool(response.get("candidate_id_valid")),
                    "action": action,
                    "selected_candidate_id": selected,
                    "ranked_candidate_ids": ranked,
                    "top1_correct": selected == int(snapshot["expected_candidate_id"]),
                    "top2_correct": int(snapshot["expected_candidate_id"]) in ranked[:2],
                    "geodesic_improved": bool(
                        selected is not None
                        and float(snapshot["candidate_ne_m"][str(selected)])
                        < float(snapshot["current_ne_m"])
                    ),
                    "parse_error": response.get("parse_error", ""),
                    "error_type": response.get("error_type", ""),
                    "metrics": response.get("metrics", {}),
                }
            )
    raw_path = output / "navigation_raw_outputs.jsonl"
    write_jsonl(raw_path, rows)
    parse_rate = sum(row["parse_ok"] for row in rows) / len(rows)
    validity = sum(row["candidate_id_valid"] for row in rows) / len(rows)
    top1 = sum(row["top1_correct"] for row in rows) / len(rows)
    top2 = sum(row["top2_correct"] for row in rows) / len(rows)
    improvement = sum(row["geodesic_improved"] for row in rows) / len(rows)
    random_baseline = sum(1.0 / row["candidate_count"] for row in rows) / len(rows)
    ordered = sorted(rows, key=lambda row: (row["episode_id"], int(row["path_index"])))
    comparisons = 0
    repeats = 0
    for previous, current in zip(ordered, ordered[1:]):
        if previous["episode_id"] == current["episode_id"]:
            comparisons += 1
            repeats += previous["selected_candidate_id"] == current["selected_candidate_id"]
    latencies = [float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in rows]
    ttft = [float(row["metrics"].get("reasoning_ttft_ms") or 0.0) for row in rows]
    decode = [float(row["metrics"].get("reasoning_tokens_per_s") or 0.0) for row in rows]
    gate = config["gate2a"]
    passed = bool(
        parse_rate >= float(gate["min_parse_rate"])
        and validity >= float(gate["min_candidate_validity"])
        and improvement > float(gate["min_geodesic_improvement_rate"])
        and top1 >= random_baseline + float(gate["min_top1_over_random"])
    )
    summary = {
        "schema_version": 2,
        "gate": "gate2a",
        "status": "PASS" if passed else "BLOCKED",
        "snapshots": len(rows),
        "json_parse_rate": parse_rate,
        "candidate_id_valid_rate": validity,
        "top1_shortest_candidate_accuracy": top1,
        "top2_shortest_candidate_accuracy": top2,
        "random_top1_baseline": random_baseline,
        "top1_over_random": top1 - random_baseline,
        "geodesic_improvement_rate": improvement,
        "repeated_candidate_rate": repeats / max(comparisons, 1),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
        "reasoning_tokens_per_s_mean": sum(decode) / len(decode),
        "raw_jsonl": str(raw_path),
        "raw_jsonl_sha256": sha256_file(raw_path),
        "rgb_source": "isaac_render_product",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if passed else 2


def main() -> int:
    config = load_yaml(args.config)
    output = Path(args.output).resolve()
    if args.mode == "prepare":
        return prepare(config, output)
    if args.mode == "collect":
        return collect(config, output)
    return evaluate(config, output)


try:
    raise SystemExit(main())
finally:
    if simulation_app is not None:
        simulation_app.close()

