#!/usr/bin/env python3
"""Gate 3B: held-out integrated NavigationPolicy + ArrivalVerifier canary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--manifest", required=True)
parser.add_argument("--scene", required=True)
parser.add_argument("--scene-usd", required=True)
parser.add_argument("--connectivity", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--run-id", default="step3_integrated_stop_canary_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.arrival_verifier import ArrivalCurrentView, ArrivalHistoryFrame, ArrivalRequest
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.evaluation import GraphEpisodeState
from step3_graph_nav.io import append_jsonl, load_yaml, read_jsonl, write_json, write_jsonl
from step3_graph_nav.isaac_scene import IsaacCandidateScene
from step3_graph_nav.protocol import GraphNavRequest
from step3_graph_nav.runner import render_candidate_views, validate_service_health


def describe_action(heading: float, distance: float) -> str:
    if heading < -25.0:
        turn = "turned right"
    elif heading > 25.0:
        turn = "turned left"
    else:
        turn = "continued forward"
    return f"{turn} by {abs(heading):.1f} degrees and moved {distance:.2f} m"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low = int(index)
    high = min(low + 1, len(values) - 1)
    part = index - low
    return values[low] * (1.0 - part) + values[high] * part


def main() -> int:
    config = load_yaml(args.config)
    gate = config["gate3b"]
    if args.scene != str(gate["scene"]):
        raise ValueError("Gate 3B scene differs from frozen config")
    allowed_ids = [int(value) for value in gate["episode_ids"]]
    episodes = [
        row for row in read_jsonl(args.manifest)
        if row.get("scene_key") == args.scene and int(row.get("episode_id")) in allowed_ids
    ]
    episodes.sort(key=lambda row: allowed_ids.index(int(row["episode_id"])))
    if [int(row["episode_id"]) for row in episodes] != allowed_ids:
        raise RuntimeError("held-out manifest does not contain frozen Gate 3B episodes")
    forbidden = set(int(value) for value in config["data"]["forbidden_development_canary"].get(args.scene, []))
    if forbidden & set(allowed_ids):
        raise RuntimeError("Gate 3B overlaps forbidden development canary")
    graph = MatterportGraph.load(args.connectivity, scene_id=args.scene)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = IsaacCandidateScene(args.scene_usd, config, str(args.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    max_calls = int(gate["max_model_calls_per_episode"])
    loop_limit = int(gate["same_node_visit_limit"])
    early_count = int(gate["false_stop_early_stop_episodes"])
    episode_rows = []
    navigation_rows = []
    arrival_rows = []
    gate_stopped = False
    gate_stop_reason = ""
    with Step3GraphNavClient(
        str(config["service"]["endpoint"]), timeout_ms=int(config["service"]["timeout_ms"])
    ) as client:
        health = client.health()
        validate_service_health(health)
        if not health.get("arrival_threshold_locked"):
            raise RuntimeError("Gate 3B requires the Gate 2B frozen arrival threshold")
        write_json(output / "service_health.json", health)
        for episode in episodes:
            state = GraphEpisodeState.from_episode(graph, episode)
            started = time.perf_counter()
            action_history: list[str] = []
            history_keyframes: list[dict[str, object]] = []
            failure_reason = "max_model_calls"
            navigation_calls = 0
            arrival_calls = 0
            episode_dir = output / "keyframes" / state.episode_id.replace(":", "__")
            for step_index in range(max_calls):
                save_dir = episode_dir / "first" if step_index == 0 else None
                specs, candidates = render_candidate_views(
                    scene,
                    graph,
                    state.current_node,
                    state.yaw_rad,
                    width=width,
                    height=height,
                    save_dir=save_dir,
                )
                if len(history_keyframes) >= 2:
                    recent_frames = history_keyframes[-4:]
                    recent_actions = action_history[-len(recent_frames) :]
                    arrival_request = ArrivalRequest(
                        episode_id=state.episode_id,
                        snapshot_id=f"{state.episode_id}:arrival3b:{step_index}:{state.current_node}",
                        instruction=state.instruction,
                        step_index=step_index,
                        current_views=tuple(
                            ArrivalCurrentView(
                                view_index=index,
                                relative_heading_deg=float(spec.relative_heading_deg),
                                jpeg=candidate.jpeg,
                                width=candidate.width,
                                height=candidate.height,
                            )
                            for index, (spec, candidate) in enumerate(zip(specs, candidates))
                        ),
                        history_frames=tuple(
                            ArrivalHistoryFrame(
                                history_index=index,
                                jpeg=frame["jpeg"],
                                width=int(frame["width"]),
                                height=int(frame["height"]),
                            )
                            for index, frame in enumerate(recent_frames)
                        ),
                        action_history=tuple(recent_actions),
                    )
                    response = client.verify_arrival(arrival_request)
                    arrival_calls += 1
                    arrival_row = {
                        "run_id": args.run_id,
                        "episode_id": state.episode_id,
                        "step_index": step_index,
                        "current_viewpoint_id": state.current_node,
                        "goal_positive": state.goal_positive,
                        "goal_ne_m": state.ne_m,
                        "raw_output": response.get("raw_output", ""),
                        "parse_ok": bool(response.get("parse_ok")),
                        "arrived": response.get("arrived"),
                        "metrics": response.get("metrics", {}),
                    }
                    arrival_rows.append(arrival_row)
                    if not response.get("parse_ok"):
                        failure_reason = "arrival_parse_failure"
                        break
                    if response.get("arrived"):
                        state.stop()
                        failure_reason = "" if state.success else "false_stop"
                        break
                nav_request = GraphNavRequest(
                    episode_id=state.episode_id,
                    snapshot_id=f"{state.episode_id}:navigate3b:{step_index}:{state.current_node}",
                    instruction=state.instruction,
                    current_viewpoint_id=state.current_node,
                    step_index=step_index,
                    candidates=candidates,
                    history=tuple(action_history[-4:]),
                )
                response = client.navigate(nav_request)
                navigation_calls += 1
                before_ne = state.ne_m
                action = response.get("action")
                nav_row = {
                    "run_id": args.run_id,
                    "episode_id": state.episode_id,
                    "step_index": step_index,
                    "current_viewpoint_id": state.current_node,
                    "goal_ne_before_m": before_ne,
                    "raw_output": response.get("raw_output", ""),
                    "parse_ok": bool(response.get("parse_ok")),
                    "candidate_id_valid": bool(response.get("candidate_id_valid")),
                    "action": action,
                    "candidate_order": [item.to_mapping() for item in specs],
                    "metrics": response.get("metrics", {}),
                }
                if not response.get("parse_ok"):
                    failure_reason = (
                        "invalid_candidate_id"
                        if response.get("error_type") == "InvalidCandidateId"
                        else "navigation_parse_failure"
                    )
                    navigation_rows.append(nav_row)
                    break
                selected_id = int(action["candidate_id"])
                selected = specs[selected_id]
                selected_image = candidates[selected_id]
                history_keyframes.append(
                    {"jpeg": selected_image.jpeg, "width": selected_image.width, "height": selected_image.height}
                )
                action_history.append(describe_action(selected.relative_heading_deg, selected.graph_distance_m))
                state.move(selected)
                nav_row.update(
                    {
                        "target_viewpoint_id": selected.target_viewpoint_id,
                        "goal_ne_after_m": state.ne_m,
                        "geodesic_improved": state.ne_m < before_ne,
                    }
                )
                navigation_rows.append(nav_row)
                if Counter(state.visited_nodes)[state.current_node] >= loop_limit:
                    failure_reason = "loop_same_node_visit_limit"
                    break
            terminal_dir = episode_dir / "terminal"
            render_candidate_views(
                scene,
                graph,
                state.current_node,
                state.yaw_rad,
                width=width,
                height=height,
                save_dir=terminal_dir,
            )
            result = state.result(
                calls=navigation_calls + arrival_calls,
                failure_reason=failure_reason,
                wall_seconds=time.perf_counter() - started,
            )
            result.update(
                {
                    "run_id": args.run_id,
                    "source_episode_id": int(episode["episode_id"]),
                    "navigation_calls": navigation_calls,
                    "arrival_calls": arrival_calls,
                    "terminal_keyframe_dir": str(terminal_dir.relative_to(output)),
                }
            )
            episode_rows.append(result)
            append_jsonl(output / "episodes.jsonl", result)
            print(json.dumps(result, separators=(",", ":")), flush=True)
            if len(episode_rows) >= early_count and all(
                row["failure_reason"] == "false_stop" for row in episode_rows[:early_count]
            ):
                gate_stopped = True
                gate_stop_reason = f"first_{early_count}_all_false_stop"
                break
    write_jsonl(output / "navigation_decisions.jsonl", navigation_rows)
    write_jsonl(output / "arrival_decisions.jsonl", arrival_rows)
    successes = sum(bool(row["success"]) for row in episode_rows)
    oracle = sum(bool(row["oracle_success"]) for row in episode_rows)
    status = "PASS" if (
        not gate_stopped
        and len(episode_rows) == len(allowed_ids)
        and successes >= int(gate["min_successes"])
    ) else "BLOCKED"
    nav_latencies = [float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in navigation_rows]
    arrival_latencies = [float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in arrival_rows]
    summary = {
        "schema_version": 2,
        "gate": "gate3b",
        "status": status,
        "episodes": len(episode_rows),
        "successes": successes,
        "success_rate": successes / max(len(episode_rows), 1),
        "oracle_successes": oracle,
        "oracle_success_rate": oracle / max(len(episode_rows), 1),
        "mean_spl": sum(float(row["spl"]) for row in episode_rows) / max(len(episode_rows), 1),
        "mean_ne_m": sum(float(row["ne_m"]) for row in episode_rows) / max(len(episode_rows), 1),
        "stop_fp": sum(int(row["stop_fp"]) for row in episode_rows),
        "stop_fn": sum(int(row["stop_fn"]) for row in episode_rows),
        "loop_episodes": sum(row["failure_reason"] == "loop_same_node_visit_limit" for row in episode_rows),
        "loop_rate": sum(row["failure_reason"] == "loop_same_node_visit_limit" for row in episode_rows)
        / max(len(episode_rows), 1),
        "navigation_parse_failure_rate": sum(not row["parse_ok"] for row in navigation_rows)
        / max(len(navigation_rows), 1),
        "arrival_parse_failure_rate": sum(not row["parse_ok"] for row in arrival_rows)
        / max(len(arrival_rows), 1),
        "invalid_candidate_ids": sum(not row["candidate_id_valid"] for row in navigation_rows),
        "navigation_calls": len(navigation_rows),
        "arrival_calls": len(arrival_rows),
        "navigation_latency_ms_p50": percentile(nav_latencies, 0.5),
        "navigation_latency_ms_p95": percentile(nav_latencies, 0.95),
        "arrival_latency_ms_p50": percentile(arrival_latencies, 0.5),
        "arrival_latency_ms_p95": percentile(arrival_latencies, 0.95),
        "episode_wall_seconds": [row["wall_seconds"] for row in episode_rows],
        "gate_stopped": gate_stopped,
        "gate_stop_reason": gate_stop_reason,
        "rgb_source": "isaac_render_product",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if status == "PASS" else 2


try:
    raise SystemExit(main())
finally:
    simulation_app.close()

