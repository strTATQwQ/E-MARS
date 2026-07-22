#!/usr/bin/env python3
"""Gate 4: run one scene of the frozen 20-episode Step3 graph-nav pilot."""

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
parser.add_argument("--run-id", default="step3_graph_nav_pilot20_v1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
from slow_benchmark.oracle_graph import MatterportGraph
from step3_graph_nav.client import Step3GraphNavClient
from step3_graph_nav.io import load_yaml, read_jsonl, write_json
from step3_graph_nav.isaac_scene import IsaacCandidateScene
from step3_graph_nav.runner import run_model_scene, summarize_run, validate_service_health


def main() -> int:
    config = load_yaml(args_cli.config)
    gate = config["gate4"]
    allowed_ids = {int(value) for value in gate["episode_ids"][args_cli.scene]}
    graph = MatterportGraph.load(args_cli.connectivity, scene_id=args_cli.scene)
    episodes = [
        row
        for row in read_jsonl(args_cli.manifest)
        if row.get("scene_key") == args_cli.scene and int(row.get("episode_id")) in allowed_ids
    ]
    if {int(row["episode_id"]) for row in episodes} != allowed_ids:
        raise ValueError(f"pilot source manifest does not contain the frozen IDs for {args_cli.scene}")
    episodes.sort(key=lambda row: gate["episode_ids"][args_cli.scene].index(int(row["episode_id"])))
    output = Path(args_cli.output).resolve()
    scene = IsaacCandidateScene(args_cli.scene_usd, config, str(args_cli.device))
    with Step3GraphNavClient(
        str(config["service"]["endpoint"]), timeout_ms=int(config["service"]["timeout_ms"])
    ) as client:
        health = client.health()
        validate_service_health(health)
        write_json(output / "service_health.json", health)
        scene_summary = run_model_scene(
            scene=scene,
            graph=graph,
            episodes=episodes,
            client=client,
            config=config,
            output=output,
            run_id=args_cli.run_id,
            gate="gate4",
            max_calls=int(gate["max_model_calls_per_episode"]),
            same_node_visit_limit=int(gate["same_node_visit_limit"]),
        )
    summary = summarize_run(output / "episodes.jsonl", output / "decisions.jsonl", run_id=args_cli.run_id)
    summary.update(
        {
            "gate": "gate4",
            "gate_stopped": scene_summary["gate_stopped"],
            "gate_stop_reason": scene_summary["gate_stop_reason"],
        }
    )
    if summary["gate_stopped"]:
        status = "BLOCKED"
    elif summary["episodes"] < int(gate["max_episodes"]):
        status = "PENDING"
    elif summary["successes"] > 0:
        status = "PASS"
    else:
        status = "BLOCKED"
    summary["status"] = status
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 2 if status == "BLOCKED" else 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
