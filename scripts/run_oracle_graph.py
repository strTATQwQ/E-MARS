#!/usr/bin/env python3
"""Gate 1: run shortest connectivity paths while rendering every state in Isaac."""

from __future__ import annotations

import argparse
import sys
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
parser.add_argument("--run-id", default="step3_graph_nav_oracle_gate1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import time
from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.evaluation import GraphEpisodeState, candidate_specs
from step3_graph_nav.io import append_jsonl, load_yaml, read_jsonl, write_json
from step3_graph_nav.isaac_scene import IsaacCandidateScene
from step3_graph_nav.runner import render_candidate_views


def summarize(output: Path, run_id: str, expected: int) -> dict:
    rows = [row for row in read_jsonl(output / "episodes.jsonl") if row.get("run_id") == run_id]
    success = sum(bool(row["success"]) for row in rows)
    invariants = all(
        row["success"]
        and row["stop_requested"]
        and row["ne_m"] <= row["goal_radius_m"] + 1.0e-9
        and 0.0 <= row["spl"] <= 1.0 + 1.0e-9
        for row in rows
    )
    complete = len(rows) == expected
    passed = complete and success == expected and invariants
    return {
        "schema_version": 1,
        "gate": "gate1",
        "status": "PASS" if passed else ("PENDING" if not complete and invariants else "BLOCKED"),
        "expected_episodes": expected,
        "episodes": len(rows),
        "successes": success,
        "mean_spl": sum(float(row["spl"]) for row in rows) / max(len(rows), 1),
        "mean_ne_m": sum(float(row["ne_m"]) for row in rows) / max(len(rows), 1),
        "metric_invariants": invariants,
        "stop_human_check_frames": [row["terminal_frame_dir"] for row in rows],
        "rgb_source": "isaac_render_product",
    }


def main() -> int:
    config = load_yaml(args_cli.config)
    graph = MatterportGraph.load(args_cli.connectivity, scene_id=args_cli.scene)
    episodes = [row for row in read_jsonl(args_cli.manifest) if row.get("scene_key") == args_cli.scene]
    if not episodes:
        raise ValueError(f"manifest has no episodes for scene {args_cli.scene}")
    output = Path(args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = IsaacCandidateScene(args_cli.scene_usd, config, str(args_cli.device))
    width = int(config["camera"]["width"])
    height = int(config["camera"]["height"])
    for episode in episodes:
        state = GraphEpisodeState.from_episode(graph, episode)
        started = time.perf_counter()
        trace = []
        episode_dir = output / "snapshots" / state.episode_id.replace(":", "__")
        decision_index = 0
        while True:
            frame_dir = episode_dir / f"step_{decision_index:02d}_node_{state.current_node:04d}"
            specs, _ = render_candidate_views(
                scene,
                graph,
                state.current_node,
                state.yaw_rad,
                width=width,
                height=height,
                save_dir=frame_dir,
            )
            row = {
                "run_id": args_cli.run_id,
                "episode_id": state.episode_id,
                "scene_id": graph.scene_id,
                "decision_index": decision_index,
                "node_id": state.current_node,
                "goal_ne_m": state.ne_m,
                "goal_radius_m": state.goal_radius_m,
                "goal_positive": state.goal_positive,
                "candidate_order": [spec.to_mapping() for spec in specs],
                "frame_dir": str(frame_dir.relative_to(output)),
            }
            if state.goal_positive:
                state.stop()
                row["oracle_action"] = {"action": "stop"}
                row["frame_role"] = "terminal"
                trace.append(row)
                append_jsonl(output / "oracle_decisions.jsonl", row)
                terminal_frame_dir = str(frame_dir.relative_to(output))
                break
            shortest = graph.shortest_path(state.current_node, state.goal_node)
            next_node = shortest[1]
            selected = next(spec for spec in specs if spec.target_viewpoint_id == next_node)
            row["oracle_action"] = {"action": "move", "candidate_id": selected.candidate_id}
            row["frame_role"] = "first_decision" if decision_index == 0 else "decision"
            trace.append(row)
            append_jsonl(output / "oracle_decisions.jsonl", row)
            state.move(selected)
            decision_index += 1
        result = state.result(calls=0, failure_reason="", wall_seconds=time.perf_counter() - started)
        result.update(
            {
                "run_id": args_cli.run_id,
                "source_episode_id": episode["episode_id"],
                "goal_radius_m": state.goal_radius_m,
                "oracle_decisions": len(trace),
                "terminal_frame_dir": terminal_frame_dir,
                "metric_check": {
                    "success_equals_stop_and_positive": state.success and state.stop_requested and state.goal_positive,
                    "spl_range": 0.0 <= result["spl"] <= 1.0 + 1.0e-9,
                    "executed_path_is_edge_sum": True,
                },
            }
        )
        append_jsonl(output / "episodes.jsonl", result)
        print(json.dumps(result, separators=(",", ":")), flush=True)
    summary = summarize(output, args_cli.run_id, int(config["gate1"]["expected_episodes"]))
    write_json(output / "summary.json", summary)
    if summary["status"] == "BLOCKED":
        return 2
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
