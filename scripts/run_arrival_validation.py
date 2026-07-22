#!/usr/bin/env python3
"""Gate 2B: build trajectory-aware arrival data, calibrate, and validate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--mode", choices=("prepare", "collect", "calibrate", "validate"), required=True)
pre_args, _ = pre.parse_known_args()

AppLauncher = None
if pre_args.mode == "collect":
    from isaaclab.app import AppLauncher as _AppLauncher

    AppLauncher = _AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("prepare", "collect", "calibrate", "validate"), required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--manifest")
parser.add_argument("--connectivity-root")
parser.add_argument("--scene")
parser.add_argument("--scene-usd")
parser.add_argument("--connectivity")
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = None
if AppLauncher is not None:
    launcher = AppLauncher(args)
    simulation_app = launcher.app

import hashlib
import json
import math
import time
import traceback
from typing import Any

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.arrival_verifier import (
    ArrivalCurrentView,
    ArrivalHistoryFrame,
    ArrivalRequest,
)
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.evaluation import GraphEpisodeState, candidate_specs
from step3_graph_nav.io import load_yaml, read_jsonl, sha256_file, write_json, write_jsonl
from step3_graph_nav.runner import render_candidate_views, validate_service_health


def require(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} is required for mode={args.mode}")
    return value


def describe_action(heading: float, distance: float) -> str:
    if heading < -25.0:
        direction = "turned right"
    elif heading > 25.0:
        direction = "turned left"
    else:
        direction = "continued forward"
    return f"{direction} by {abs(heading):.1f} degrees and moved {distance:.2f} m"


def episode_states(graph: MatterportGraph, episode: dict[str, Any], history_max: int) -> list[dict[str, Any]]:
    state = GraphEpisodeState.from_episode(graph, episode)
    nodes = [state.current_node]
    yaws = [state.yaw_rad]
    actions: list[str] = []
    while not state.goal_positive:
        next_node = graph.shortest_path(state.current_node, state.goal_node)[1]
        spec = next(
            item for item in candidate_specs(graph, state.current_node, state.yaw_rad)
            if item.target_viewpoint_id == next_node
        )
        actions.append(describe_action(spec.relative_heading_deg, spec.graph_distance_m))
        state.move(spec)
        nodes.append(state.current_node)
        yaws.append(state.yaw_rad)
    terminal_index = len(nodes) - 1
    if terminal_index < 3:
        return []
    negative_index = max(2, terminal_index // 2)
    negative_index = min(negative_index, terminal_index - 1)
    result = []
    for arrived, state_index in ((False, negative_index), (True, terminal_index)):
        history_start = max(0, state_index - history_max)
        history_indices = list(range(history_start, state_index))
        if len(history_indices) < 2:
            return []
        specs = candidate_specs(graph, nodes[state_index], yaws[state_index])
        result.append(
            {
                "scene_id": graph.scene_id,
                "source_episode_id": int(episode["episode_id"]),
                "episode_id": str(episode["benchmark_episode_id"]),
                "instruction": str(episode["instruction"]["instruction_text"]),
                "split_role": "",
                "arrived_label": arrived,
                "step_index": state_index,
                "current_viewpoint_id": nodes[state_index],
                "current_yaw_rad": yaws[state_index],
                "current_candidate_order": [item.to_mapping() for item in specs],
                "history_viewpoint_ids": [nodes[index] for index in history_indices],
                "history_yaw_rad": [yaws[index] for index in history_indices],
                "action_history": [actions[index] for index in history_indices],
            }
        )
    return result


def prepare(config: dict[str, Any], output: Path) -> int:
    manifest = Path(require(args.manifest, "--manifest")).resolve()
    connectivity_root = Path(require(args.connectivity_root, "--connectivity-root")).resolve()
    source = read_jsonl(manifest)
    gate = config["gate2b"]
    history_max = int(gate["history_max_frames"])
    plans = []
    selected_episode_keys: dict[str, list[str]] = {}
    graph_hashes = {}
    for role, scene_key in (
        ("calibration", "calibration_scenes"),
        ("validation", "validation_scenes"),
    ):
        allowed = set(str(item) for item in gate[scene_key])
        eligible = []
        for episode in source:
            scene_id = str(episode.get("scene_key"))
            if scene_id not in allowed:
                continue
            graph_path = connectivity_root / f"{scene_id}_connectivity.json"
            graph_hashes[scene_id] = sha256_file(graph_path)
            graph = MatterportGraph.load(graph_path, scene_id=scene_id)
            rows = episode_states(graph, episode, history_max)
            if len(rows) == 2:
                eligible.append(rows)
        needed = int(gate["episodes_per_split"])
        if len(eligible) < needed:
            raise RuntimeError(f"{role} has only {len(eligible)} eligible episodes; need {needed}")
        selected = eligible[:needed]
        selected_episode_keys[role] = [rows[0]["episode_id"] for rows in selected]
        for rows in selected:
            for row in rows:
                row["split_role"] = role
                plans.append(row)
    calibration_ids = set(selected_episode_keys["calibration"])
    validation_ids = set(selected_episode_keys["validation"])
    if calibration_ids & validation_ids:
        raise RuntimeError("arrival calibration and validation episodes overlap")
    for index, row in enumerate(plans):
        row["state_index"] = index
        label = "arrived" if row["arrived_label"] else "not_arrived"
        row["snapshot_id"] = (
            f"arrival2b:{row['split_role']}:{index:02d}:{row['scene_id']}:"
            f"{row['source_episode_id']}:{row['step_index']}:{label}"
        )
    expected_per_split = int(gate["states_per_split"])
    for role in ("calibration", "validation"):
        rows = [row for row in plans if row["split_role"] == role]
        if len(rows) != expected_per_split:
            raise RuntimeError(f"{role} produced {len(rows)} states; expected {expected_per_split}")
        if sum(bool(row["arrived_label"]) for row in rows) != int(gate["arrived_per_split"]):
            raise RuntimeError(f"{role} arrived count mismatch")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "state_plan.jsonl"
    write_jsonl(plan_path, plans)
    frozen = {
        "schema_version": 2,
        "gate": "gate2b",
        "status": "FROZEN_BEFORE_MODEL_EVALUATION",
        "source_split": "train",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "connectivity_sha256": graph_hashes,
        "state_plan_sha256": sha256_file(plan_path),
        "states_total": len(plans),
        "arrived_total": sum(bool(row["arrived_label"]) for row in plans),
        "not_arrived_total": sum(not bool(row["arrived_label"]) for row in plans),
        "calibration_episode_ids": selected_episode_keys["calibration"],
        "validation_episode_ids": selected_episode_keys["validation"],
        "episode_overlap": [],
        "history_frames_min": min(len(row["history_viewpoint_ids"]) for row in plans),
        "history_frames_max": max(len(row["history_viewpoint_ids"]) for row in plans),
    }
    write_json(output / "state_plan_manifest.json", frozen)
    print(json.dumps(frozen, separators=(",", ":")), flush=True)
    return 0


def _yaw_key(yaw: float) -> str:
    return hashlib.sha256(f"{yaw:.12f}".encode("ascii")).hexdigest()[:10]


def collect(config: dict[str, Any], output: Path) -> int:
    from step3_graph_nav.isaac_scene import IsaacCandidateScene

    scene_id = require(args.scene, "--scene")
    scene_usd = require(args.scene_usd, "--scene-usd")
    connectivity = require(args.connectivity, "--connectivity")
    graph = MatterportGraph.load(connectivity, scene_id=scene_id)
    plan = [row for row in read_jsonl(output / "state_plan.jsonl") if row["scene_id"] == scene_id]
    if not plan:
        raise ValueError(f"arrival plan has no states for scene {scene_id}")
    scene = IsaacCandidateScene(scene_usd, config, str(args.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    cache_dir = output / "frame_cache" / scene_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    panorama_cache: dict[tuple[int, float], list[dict[str, Any]]] = {}
    history_cache: dict[tuple[int, float], dict[str, Any]] = {}
    collected = []
    current_render_count = 0
    history_render_count = 0
    for row in plan:
        current_yaw = float(row["current_yaw_rad"])
        current_key = (int(row["current_viewpoint_id"]), round(current_yaw, 10))
        if current_key not in panorama_cache:
            frame_dir = cache_dir / f"current_{current_key[0]:04d}_{_yaw_key(current_key[1])}"
            expected_files = [
                frame_dir / f"candidate_{int(item['candidate_id']):02d}.jpg"
                for item in row["current_candidate_order"]
            ]
            if expected_files and all(path.exists() for path in expected_files):
                panorama_cache[current_key] = [
                    {
                        "view_index": int(item["candidate_id"]),
                        "relative_heading_deg": float(item["relative_heading_deg"]),
                        "width": width,
                        "height": height,
                        "rgb_file": str(path.relative_to(output)),
                        "rgb_sha256": sha256_file(path),
                    }
                    for item, path in zip(row["current_candidate_order"], expected_files)
                ]
            else:
                specs, candidates = render_candidate_views(
                    scene,
                    graph,
                    current_key[0],
                    current_yaw,
                    width=width,
                    height=height,
                    save_dir=frame_dir,
                )
                if [item.to_mapping() for item in specs] != row["current_candidate_order"]:
                    raise RuntimeError(f"current panorama order drift for {row['snapshot_id']}")
                panorama_cache[current_key] = [
                    {
                        "view_index": candidate.candidate_id,
                        "relative_heading_deg": candidate.relative_heading_deg,
                        "width": candidate.width,
                        "height": candidate.height,
                        "rgb_file": str(
                            (frame_dir / f"candidate_{candidate.candidate_id:02d}.jpg").relative_to(output)
                        ),
                        "rgb_sha256": hashlib.sha256(candidate.jpeg).hexdigest(),
                    }
                    for candidate in candidates
                ]
                current_render_count += len(candidates)
        history_images = []
        for history_index, (node_id, yaw) in enumerate(
            zip(row["history_viewpoint_ids"], row["history_yaw_rad"])
        ):
            key = (int(node_id), round(float(yaw), 10))
            if key not in history_cache:
                target = cache_dir / f"history_{key[0]:04d}_{_yaw_key(key[1])}.jpg"
                if target.exists():
                    jpeg = target.read_bytes()
                else:
                    print(
                        json.dumps(
                            {"event": "render_history_keyframe", "scene_id": scene_id, "node_id": key[0]},
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    jpeg, _, _ = scene.render(graph.nodes[key[0]].camera_position, float(yaw))
                    target.write_bytes(jpeg)
                    history_render_count += 1
                history_cache[key] = {
                    "width": width,
                    "height": height,
                    "rgb_file": str(target.relative_to(output)),
                    "rgb_sha256": hashlib.sha256(jpeg).hexdigest(),
                }
            history_images.append({"history_index": history_index, **history_cache[key]})
        value = dict(row)
        value.update(
            {
                "scene_usd": str(Path(scene_usd).resolve()),
                "scene_usd_sha256": sha256_file(scene_usd),
                "connectivity": str(Path(connectivity).resolve()),
                "rgb_source": "isaac_render_product",
                "current_images": panorama_cache[current_key],
                "history_images": history_images,
            }
        )
        collected.append(value)
    states_path = output / "states.jsonl"
    existing = []
    if states_path.exists():
        existing = [row for row in read_jsonl(states_path) if row["scene_id"] != scene_id]
    merged = sorted(existing + collected, key=lambda row: int(row["state_index"]))
    write_jsonl(states_path, merged)
    report = {
        "schema_version": 2,
        "gate": "gate2b",
        "scene_id": scene_id,
        "scene_state_count": len(collected),
        "total_collected": len(merged),
        "current_candidate_renders": current_render_count,
        "unique_history_renders": history_render_count,
        "rgb_source": "isaac_render_product",
    }
    write_json(output / f"collection_{scene_id}.json", report)
    print(json.dumps(report, separators=(",", ":")), flush=True)
    return 0


def make_request(output: Path, state: dict[str, Any]) -> ArrivalRequest:
    current = tuple(
        ArrivalCurrentView(
            view_index=int(item["view_index"]),
            relative_heading_deg=float(item["relative_heading_deg"]),
            jpeg=(output / item["rgb_file"]).read_bytes(),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in state["current_images"]
    )
    history = tuple(
        ArrivalHistoryFrame(
            history_index=int(item["history_index"]),
            jpeg=(output / item["rgb_file"]).read_bytes(),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in state["history_images"]
    )
    return ArrivalRequest(
        episode_id=str(state["episode_id"]),
        snapshot_id=str(state["snapshot_id"]),
        instruction=str(state["instruction"]),
        step_index=int(state["step_index"]),
        current_views=current,
        history_frames=history,
        action_history=tuple(str(item) for item in state["action_history"]),
    )


def classification_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum((not label) and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and (not prediction) for label, prediction in zip(labels, predictions))
    tn = sum((not label) and (not prediction) for label, prediction in zip(labels, predictions))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def choose_threshold(labels: list[bool], margins: list[float]) -> tuple[float, dict[str, Any]]:
    unique = sorted(set(margins))
    thresholds = [unique[0] - 1.0]
    thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    thresholds.append(unique[-1] + 1.0)
    scored = []
    for threshold in thresholds:
        metrics = classification_metrics(labels, [margin >= threshold for margin in margins])
        scored.append((metrics["balanced_accuracy"], metrics["f1"], metrics["precision"], threshold, metrics))
    _, _, _, threshold, metrics = max(scored, key=lambda item: (item[0], item[1], item[2], item[3]))
    return float(threshold), metrics


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    part = index - low
    return values[low] * (1.0 - part) + values[high] * part


def run_split(config: dict[str, Any], output: Path, role: str) -> int:
    states = [row for row in read_jsonl(output / "states.jsonl") if row["split_role"] == role]
    expected = int(config["gate2b"]["states_per_split"])
    if len(states) != expected:
        raise RuntimeError(f"{role} requires {expected} states, found {len(states)}")
    rows = []
    threshold_path = output / "calibration_threshold.json"
    with Step3GraphNavClient(
        str(config["service"]["endpoint"]), timeout_ms=int(config["service"]["timeout_ms"])
    ) as client:
        health = client.health()
        validate_service_health(health)
        if role == "validation":
            threshold_value = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
            if not health.get("arrival_threshold_locked") or not math.isclose(
                float(health.get("arrival_threshold")), float(threshold_value), abs_tol=1.0e-9
            ):
                raise RuntimeError("validation service does not hold the frozen calibration threshold")
        write_json(output / f"service_health_{role}.json", health)
        for state in states:
            response = client.verify_arrival(make_request(output, state))
            rows.append(
                {
                    "state_index": state["state_index"],
                    "snapshot_id": state["snapshot_id"],
                    "episode_id": state["episode_id"],
                    "arrived_label": bool(state["arrived_label"]),
                    "raw_output": response.get("raw_output", ""),
                    "parse_ok": bool(response.get("parse_ok")),
                    "arrived": response.get("arrived"),
                    "parse_error": response.get("parse_error", ""),
                    "margin": float(response.get("metrics", {}).get("arrived_margin")),
                    "metrics": response.get("metrics", {}),
                }
            )
        if role == "calibration":
            labels = [row["arrived_label"] for row in rows]
            margins = [row["margin"] for row in rows]
            threshold, calibrated_metrics = choose_threshold(labels, margins)
            for row in rows:
                row["calibrated_prediction"] = row["margin"] >= threshold
            freeze_response = client.set_arrival_threshold(threshold)
            threshold_value = {
                "schema_version": 2,
                "threshold": threshold,
                "selection_split": "train_calibration",
                "selection_objective": "balanced_accuracy_then_f1_then_precision",
                "calibration_metrics": calibrated_metrics,
                "service_response": freeze_response,
            }
            write_json(threshold_path, threshold_value)
    raw_path = output / f"{role}_raw_outputs.jsonl"
    write_jsonl(raw_path, rows)
    parse_rate = sum(row["parse_ok"] for row in rows) / len(rows)
    if role == "calibration":
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
        predictions = [row["margin"] >= threshold for row in rows]
    else:
        predictions = [bool(row["arrived"]) for row in rows]
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"]
    labels = [row["arrived_label"] for row in rows]
    quality = classification_metrics(labels, predictions)
    latencies = [float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in rows]
    ttft = [float(row["metrics"].get("reasoning_ttft_ms") or 0.0) for row in rows]
    decode = [float(row["metrics"].get("reasoning_tokens_per_s") or 0.0) for row in rows]
    arrived_margins = [row["margin"] for row in rows if row["arrived_label"]]
    not_arrived_margins = [row["margin"] for row in rows if not row["arrived_label"]]
    summary = {
        "schema_version": 2,
        "gate": "gate2b",
        "split_role": role,
        "states": len(rows),
        "threshold": threshold,
        "json_parse_rate": parse_rate,
        **quality,
        "arrived_margin_distribution": {
            "values": arrived_margins,
            "min": min(arrived_margins),
            "median": percentile(arrived_margins, 0.5),
            "max": max(arrived_margins),
        },
        "not_arrived_margin_distribution": {
            "values": not_arrived_margins,
            "min": min(not_arrived_margins),
            "median": percentile(not_arrived_margins, 0.5),
            "max": max(not_arrived_margins),
        },
        "ttft_ms_p50": percentile(ttft, 0.5),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "reasoning_tokens_per_s_mean": sum(decode) / len(decode),
        "raw_jsonl": str(raw_path),
        "raw_jsonl_sha256": sha256_file(raw_path),
        "rgb_source": "isaac_render_product",
    }
    if role == "validation":
        gate = config["gate2b"]
        passed = bool(
            parse_rate >= 0.95
            and quality["precision"] >= float(gate["min_precision"])
            and quality["recall"] >= float(gate["min_recall"])
            and quality["balanced_accuracy"] >= float(gate["min_balanced_accuracy"])
        )
        summary["status"] = "PASS" if passed else "BLOCKED"
    else:
        passed = True
        summary["status"] = "CALIBRATED"
    write_json(output / f"summary_{role}.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if passed else 2


def main() -> int:
    config = load_yaml(args.config)
    output = Path(args.output).resolve()
    if args.mode == "prepare":
        return prepare(config, output)
    if args.mode == "collect":
        return collect(config, output)
    return run_split(config, output, "calibration" if args.mode == "calibrate" else "validation")


exit_code = 0
try:
    exit_code = main()
except BaseException:
    traceback.print_exc()
    exit_code = 1
finally:
    if simulation_app is not None:
        simulation_app.close()
raise SystemExit(exit_code)
