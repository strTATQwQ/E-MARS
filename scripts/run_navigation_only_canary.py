#!/usr/bin/env python3
"""Gate 3A: move-only held-out canary with runner auto-stop in success radius."""

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
parser.add_argument("--run-id", default="step3_navigation_only_canary_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.evaluation import GraphEpisodeState
from step3_graph_nav.io import append_jsonl, load_yaml, read_jsonl, write_json
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
    gate = config["gate3a"]
    if args.scene != str(gate["scene"]):
        raise ValueError("Gate 3A scene differs from frozen config")
    allowed_ids = [int(value) for value in gate["episode_ids"]]
    episodes = [
        row for row in read_jsonl(args.manifest)
        if row.get("scene_key") == args.scene and int(row.get("episode_id")) in allowed_ids
    ]
    episodes.sort(key=lambda row: allowed_ids.index(int(row["episode_id"])))
    if [int(row["episode_id"]) for row in episodes] != allowed_ids:
        raise RuntimeError("held-out manifest does not contain frozen Gate 3A episodes")
    forbidden = set(int(value) for value in config["data"]["forbidden_development_canary"].get(args.scene, []))
    if forbidden & set(allowed_ids):
        raise RuntimeError("Gate 3A overlaps forbidden development canary")
    graph = MatterportGraph.load(args.connectivity, scene_id=args.scene)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = IsaacCandidateScene(args.scene_usd, config, str(args.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    max_calls = int(gate["max_model_calls_per_episode"])
    loop_limit = int(gate["same_node_visit_limit"])
    early_limit = int(gate["early_stop_after_oracle_failures"])
    episode_rows = []
    decision_rows = []
    gate_stopped = False
    gate_stop_reason = ""
    with Step3GraphNavClient(
        str(config["service"]["endpoint"]), timeout_ms=int(config["service"]["timeout_ms"])
    ) as client:
        health = client.health()
        validate_service_health(health)
        write_json(output / "service_health.json", health)
        for episode in episodes:
            state = GraphEpisodeState.from_episode(graph, episode)
            started = time.perf_counter()
            history: list[str] = []
            failure_reason = "max_model_calls"
            calls = 0
            improvement_steps = 0
            episode_dir = output / "keyframes" / state.episode_id.replace(":", "__")
            if state.goal_positive:
                state.stop()
                failure_reason = ""
            for step_index in range(max_calls):
                if state.stop_requested:
                    break
                first_dir = episode_dir / "first" if step_index == 0 else None
                specs, candidates = render_candidate_views(
                    scene,
                    graph,
                    state.current_node,
                    state.yaw_rad,
                    width=width,
                    height=height,
                    save_dir=first_dir,
                )
                request = GraphNavRequest(
                    episode_id=state.episode_id,
                    snapshot_id=f"{state.episode_id}:nav3a:{step_index}:{state.current_node}",
                    instruction=state.instruction,
                    current_viewpoint_id=state.current_node,
                    step_index=step_index,
                    candidates=candidates,
                    history=tuple(history[-4:]),
                )
                response = client.navigate(request)
                calls += 1
                before_ne = state.ne_m
                action = response.get("action")
                row = {
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
                        else "parse_failure"
                    )
                    row["failure_reason"] = failure_reason
                    decision_rows.append(row)
                    break
                selected = specs[int(action["candidate_id"])]
                state.move(selected)
                after_ne = state.ne_m
                improved = after_ne < before_ne
                improvement_steps += int(improved)
                row.update(
                    {
                        "target_viewpoint_id": selected.target_viewpoint_id,
                        "goal_ne_after_m": after_ne,
                        "geodesic_improved": improved,
                    }
                )
                decision_rows.append(row)
                history.append(describe_action(selected.relative_heading_deg, selected.graph_distance_m))
                if state.goal_positive:
                    state.stop()
                    failure_reason = ""
                    break
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
                calls=calls,
                failure_reason=failure_reason,
                wall_seconds=time.perf_counter() - started,
            )
            result.update(
                {
                    "run_id": args.run_id,
                    "source_episode_id": int(episode["episode_id"]),
                    "auto_stop_success": bool(state.success),
                    "spl_like": float(result["spl"]),
                    "geodesic_progress_m": float(result["shortest_path_m"] - result["ne_m"]),
                    "geodesic_improvement_rate": improvement_steps / max(calls, 1),
                    "terminal_keyframe_dir": str(terminal_dir.relative_to(output)),
                }
            )
            episode_rows.append(result)
            append_jsonl(output / "episodes.jsonl", result)
            print(json.dumps(result, separators=(",", ":")), flush=True)
            if len(episode_rows) >= early_limit and not any(
                row["oracle_success"] for row in episode_rows[:early_limit]
            ):
                gate_stopped = True
                gate_stop_reason = f"first_{early_limit}_oracle_success_all_zero"
                break
    write_jsonl(output / "decisions.jsonl", decision_rows)
    latencies = [float(row["metrics"].get("client_roundtrip_ms") or 0.0) for row in decision_rows]
    ttft = [float(row["metrics"].get("reasoning_ttft_ms") or 0.0) for row in decision_rows]
    decode = [float(row["metrics"].get("reasoning_tokens_per_s") or 0.0) for row in decision_rows]
    successes = sum(bool(row["auto_stop_success"]) for row in episode_rows)
    oracle = sum(bool(row["oracle_success"]) for row in episode_rows)
    status = "PASS" if (
        not gate_stopped
        and len(episode_rows) == len(allowed_ids)
        and oracle >= int(gate["min_oracle_successes"])
    ) else "BLOCKED"
    summary = {
        "schema_version": 2,
        "gate": "gate3a",
        "status": status,
        "episodes": len(episode_rows),
        "auto_stop_successes": successes,
        "auto_stop_success_rate": successes / max(len(episode_rows), 1),
        "oracle_successes": oracle,
        "oracle_success_rate": oracle / max(len(episode_rows), 1),
        "mean_ne_m": sum(float(row["ne_m"]) for row in episode_rows) / max(len(episode_rows), 1),
        "mean_spl_like": sum(float(row["spl_like"]) for row in episode_rows) / max(len(episode_rows), 1),
        "mean_geodesic_progress_m": sum(float(row["geodesic_progress_m"]) for row in episode_rows)
        / max(len(episode_rows), 1),
        "mean_geodesic_improvement_rate": sum(float(row["geodesic_improvement_rate"]) for row in episode_rows)
        / max(len(episode_rows), 1),
        "loop_episodes": sum(row["failure_reason"] == "loop_same_node_visit_limit" for row in episode_rows),
        "loop_rate": sum(row["failure_reason"] == "loop_same_node_visit_limit" for row in episode_rows)
        / max(len(episode_rows), 1),
        "parse_failure_rate": sum(not row["parse_ok"] for row in decision_rows)
        / max(len(decision_rows), 1),
        "invalid_candidate_ids": sum(not row["candidate_id_valid"] for row in decision_rows),
        "model_calls": len(decision_rows),
        "ttft_ms_p50": percentile(ttft, 0.5),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "latency_ms_p50": percentile(latencies, 0.5),
        "latency_ms_p95": percentile(latencies, 0.95),
        "reasoning_tokens_per_s_mean": sum(decode) / max(len(decode), 1),
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

